import random
import re
from itertools import combinations
from queue import Queue

from tree_sitter import Language, Parser


class TSNode:
    def __init__(self):
        self.type = None
        self.value = None
        self.parent = None
        self.children = list()

    def gen_root_path(self, tree_style='AST'):
        ptr = self
        root_path = []
        hpt_ptr = None
        while ptr.parent:
            ptr = ptr.parent
            # AST | SPT || HST | HPT
            if tree_style == 'AST':
                # abstract syntax tree
                root_path.append(ptr.type)
            elif tree_style == 'SPT':
                # simplified parse tree
                root_path.append(ptr.value)
            elif tree_style == 'HST':
                # binary syntax tree
                # hierarchy syntax tree
                root_path.append(ptr.type)
                if not hpt_ptr and ptr.type == 'expression_statement':
                    hpt_ptr = ptr
                    root_path = []
            elif tree_style == 'HPT':
                # binary parse tree
                # hierarchy parse tree
                root_path.append(ptr.value)
                if not hpt_ptr and ptr.type == 'expression_statement':
                    hpt_ptr = ptr
                    root_path = []
        root_path = list(reversed(root_path))
        value = self.value
        if hpt_ptr:
            values = list()
            q = Queue()
            q.put(hpt_ptr)
            while not q.empty():
                ptr = q.get()
                if ptr.type == 'identifier':
                    values.extend(ptr.value.split('|'))
                for child in ptr.children:
                    q.put(child)
            value = '|'.join(set(values))
        return root_path, value

    def gen_sbt(self):
        sbt4children = ''.join([child.gen_sbt() for child in self.children])
        # we prefer self.value to self.type
        # return f'({self.type}{sbt4children}){self.type}'
        return f'({self.value}{sbt4children}){self.value}'


class TS:
    def __init__(self, code, language='python', tree_style='SPT', path_style='L2L'):
        # AST | SPT || HST | HPT
        self.tree_style = tree_style
        # L2L | UD | U2D
        self.path_style = path_style
        # Use the Language.build_library method to compile these
        # into a library that's usable from Python:
        csn_so = 'build/csn.so'
        # Language.build_library(
        #   csn_so,
        #   [
        #     'vendor/tree-sitter-go',
        #     'vendor/tree-sitter-java',
        #     'vendor/tree-sitter-javascript',
        #     'vendor/tree-sitter-php',
        #     'vendor/tree-sitter-python',
        #     'vendor/tree-sitter-ruby',
        #   ]
        # )
        parser = Parser()
        # Load the languages into your app as Language objects:
        # ('go', 'java', 'javascript', 'php', 'python', 'ruby')
        parser.set_language(Language(csn_so, language))
        tree = parser.parse(code.encode())
        code_lines = code.split('\n')
        self.root, self.terminals = self.traverse(tree, code_lines)
        self.debug = False
        if self.debug:
            print(f'{"@" * 9}code\n{code}')
            print(f'{"@" * 9}sexp\n{tree.root_node.sexp()}')

    def traverse(self, tree, code_lines):
        q = Queue()
        root = TSNode()
        terminals = list()
        q.put((root, tree.root_node))
        while not q.empty():
            # lhs is the node we defined
            # rhs is the node TS supplied
            lhs, rhs = q.get()
            lhs.type = self.normalize(rhs.type)
            lhs.value = self.query_token(rhs, code_lines)
            if rhs.children:
                lhs.value = self.simplify(lhs.value)
                for rhs_child in rhs.children:
                    lhs_child = TSNode()
                    lhs_child.parent = lhs
                    lhs.children.append(lhs_child)
                    q.put((lhs_child, rhs_child))
            else:
                lhs.value = self.tokenize(lhs.value)
                terminals.append(lhs)
        return root, terminals

    def gen_identifiers(self):
        # it is to generate the sequence of leaf nodes
        # it performs better than the set of identifiers
        identifiers = list()
        for terminal in self.terminals:
            if terminal.type == 'identifier':
                identifier = terminal.value
                identifiers.append(identifier)
        return identifiers

    def gen_root_paths(self):
        root_paths = list()

        for terminal in self.terminals:
            if terminal.type == 'identifier':
                root_path, value = terminal.gen_root_path(self.tree_style)
                if root_path and value:
                    root_paths.append((root_path, value))

        if self.debug:
            print(f'{"@" * 9}root_paths\n{root_paths}')
        return root_paths

    def gen_tree_paths(self):
        root_paths = self.gen_root_paths()
        tree_paths = []
        max_path_length = 8
        max_path_width = 2
        cases = list(combinations(iterable=root_paths, r=2))
        # JGD maybe consider some sampling strategies
        cases = random.sample(cases, min(20, len(cases)))
        for (u_path, u_value), (v_path, v_value) in cases:
            prefix, lca, suffix = self.merge_paths(u_path, v_path)
            prefix_len = len(prefix)
            suffix_len = len(suffix)
            # 1 <= prefix_len and 1 <= suffix_len
            if prefix_len + 1 + suffix_len <= max_path_length\
                    and abs(prefix_len - suffix_len) <= max_path_width:
                source, target = u_value, v_value
                if self.path_style == 'L2L':
                    middle = '|'.join(prefix + [lca] + suffix)
                elif self.path_style == 'UD':
                    middle = '|'.join('U' * prefix_len + 'D' * suffix_len)
                else:
                    middle = '|U|'.join(prefix) + f'|U|{lca}|D|' + '|D|'.join(suffix)
                # tree_path = middle
                tree_path = f'{source}|{middle}|{target}'
                tree_paths.append(tree_path)

        if self.debug:
            print(f'{"@" * 9}tree_paths\n{tree_paths}')
        return tree_paths

    @staticmethod
    def query_token(node, code_lines):
        line_start = node.start_point[0]
        line_end = node.end_point[0]
        char_start = node.start_point[1]
        char_end = node.end_point[1]

        if line_start != line_end:
            token = code_lines[line_start][char_start:]
        else:
            token = code_lines[line_start][char_start:char_end]
        return token

    @staticmethod
    def simplify(token):
        for keyword in ('if', 'else', 'elif', 'switch', 'case', 'default'):
            if keyword in token:
                return 'case'
        for keyword in ('for', 'while', 'do', 'break', 'continue'):
            if keyword in token:
                return 'loop'
        for keyword in ('{', '}'):
            if keyword in token:
                return 'block'
        for keyword in ('void', 'protected', 'public', 'private', 'function', 'func', 'def'):
            if keyword in token:
                return 'method'
        if '=' in token:
            return 'assign'
        elif '.' in token:
            return 'field'
        elif '(' in token or ')' in token:
            return 'invoke'
        elif '[' in token or ']' in token:
            return 'access'
        else:
            return 'literal'

    @staticmethod
    def normalize(term):
        return term.lower()

    @staticmethod
    def tokenize(term):
        def camel_case_split(identifier):
            matches = re.finditer(
                '.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)',
                identifier,
            )
            return [m.group(0) for m in matches]

        blocks = []
        for underscore_block in term.split('_'):
            blocks.extend(camel_case_split(underscore_block))

        return '|'.join(block.lower() for block in blocks)

    @staticmethod
    def merge_paths(u_path, v_path):
        m, n, s = len(u_path), len(v_path), 0
        while s < min(m, n) and u_path[s] == v_path[s]:
            s += 1

        prefix = list(reversed(u_path[s:]))
        lca = u_path[s - 1]
        suffix = v_path[s:]

        return prefix, lca, suffix

    def gen_sbt_representation(self):
        sbt_representation = self.root.gen_sbt()
        sbt_tokens = re.split('[()]', sbt_representation)
        sbt_tokens = list(filter(None, sbt_tokens))
        return sbt_tokens


def code2identifiers(code, language='python'):
    ts = TS(code, language)
    identifiers = ts.gen_identifiers()
    return identifiers


def code2paths(code, language='python'):
    ts = TS(code, language)
    paths = ts.gen_tree_paths()
    return paths


def code2sbt(code, language='python'):
    ts = TS(code, language)
    sbt_tokens = ts.gen_sbt_representation()
    return sbt_tokens


'''
go@@@@@@@@@code
func (s *SkuM1Small) GetInnkeeperClient() (innkeeperclient.InnkeeperClient, error) {
        var err error
        if s.Client == nil {
                if clnt, err := s.InitInnkeeperClient(); err == nil {
                        s.Client = clnt
                } else {
                        lo.G.Error("error parsing current cfenv: ", err.Error())
                }
        }
        return s.Client, err
}
go@@@@@@@@@sexp
(source_file (method_declaration receiver: (parameter_list (parameter_declaration name: (identifier) type: (pointer_type (type_identifier)))) name: (field_identifier) parameters: (parameter_list) result: (parameter_list (parameter_declaration type: (qualified_type package: 
(package_identifier) name: (type_identifier))) (parameter_declaration type: (type_identifier))) body: (block (var_declaration (var_spec name: (identifier) type: (type_identifier))) (if_statement condition: (binary_expression left: (selector_expression operand: (identifier) 
field: (field_identifier)) right: (nil)) consequence: (block (if_statement initializer: (short_var_declaration left: (expression_list (identifier) (identifier)) right: (expression_list (call_expression function: (selector_expression operand: (identifier) field: (field_identifier)) arguments: (argument_list)))) condition: (binary_expression left: (identifier) right: (nil)) consequence: (block (assignment_statement left: (expression_list (selector_expression operand: (identifier) field: (field_identifier))) right: (expression_list (identifier)))) alternative: (block (call_expression function: (selector_expression operand: (selector_expression operand: (identifier) field: (field_identifier)) field: (field_identifier)) arguments: (argument_list (interpreted_string_literal) (call_expression function: (selector_expression operand: (identifier) field: (field_identifier)) arguments: (argument_list)))))))) (return_statement (expression_list (selector_expression operand: (identifier) field: (field_identifier)) (identifier))))))
java@@@@@@@@@code
protected void notifyAttemptToReconnectIn(int seconds) {
        if (isReconnectionAllowed()) {
            for (ConnectionListener listener : connection.connectionListeners) {
                listener.reconnectingIn(seconds);
            }
        }
    }
java@@@@@@@@@sexp
(program (local_variable_declaration (modifiers) type: (void_type) declarator: (variable_declarator name: (identifier)) (MISSING ";")) (ERROR (formal_parameters (formal_parameter type: (integral_type) name: (identifier)))) (block (if_statement condition: (parenthesized_expression (method_invocation name: (identifier) arguments: (argument_list))) consequence: (block (enhanced_for_statement type: (type_identifier) name: (identifier) value: (scoped_identifier scope: (identifier) name: (identifier)) body: (block (expression_statement (method_invocation object: (identifier) name: (identifier) arguments: (argument_list (identifier))))))))))
javascript@@@@@@@@@code
function (context, grunt) {
  this.context = context;
  this.grunt = grunt;

  // Merge task-specific and/or target-specific options with these defaults.
  this.options = context.options(defaultOptions);
}
javascript@@@@@@@@@sexp
(program (expression_statement (function parameters: (formal_parameters (identifier) (identifier)) body: (statement_block (expression_statement (assignment_expression left: (member_expression object: (this) property: (property_identifier)) right: (identifier))) (expression_statement (assignment_expression left: (member_expression object: (this) property: (property_identifier)) right: (identifier))) (comment) (expression_statement (assignment_expression left: (member_expression object: (this) property: (property_identifier)) right: (call_expression function: (member_expression object: (identifier) property: (property_identifier)) arguments: (arguments (identifier)))))))))      
php@@@@@@@@@code
public function init()
    {
        parent::init();
        if ($this->message === null) {
            $this->message = \Reaction::t('rct', '{attribute} is invalid.');
        }
    }
php@@@@@@@@@sexp
(program (text))
python@@@@@@@@@code
def get_url_args(url):
    """ Returns a dictionary from a URL params """
    url_data = urllib.parse.urlparse(url)
    arg_dict = urllib.parse.parse_qs(url_data.query)
    return arg_dict
python@@@@@@@@@sexp
(module (function_definition name: (identifier) parameters: (parameters (identifier)) body: (block (expression_statement (string)) (expression_statement (assignment left: (expression_list (identifier)) right: (expression_list (call function: (attribute object: (attribute object: (identifier) attribute: (identifier)) attribute: (identifier)) arguments: (argument_list (identifier)))))) (expression_statement (assignment left: (expression_list (identifier)) right: (expression_list (call function: (attribute object: (attribute object: (identifier) attribute: (identifier)) attribute: (identifier)) arguments: (argument_list (attribute object: (identifier) attribute: (identifier))))))) (return_statement (expression_list (identifier))))))
ruby@@@@@@@@@code
def part(name)
      parts.select {|p| p.name.downcase == name.to_s.downcase }.first
    end
ruby@@@@@@@@@sexp
(program (method name: (identifier) parameters: (method_parameters (identifier)) (call receiver: (method_call method: (call receiver: (identifier) method: (identifier)) block: (block (block_parameters (identifier)) (binary left: (call receiver: (call receiver: (identifier) 
method: (identifier)) method: (identifier)) right: (call receiver: (call receiver: (identifier) method: (identifier)) method: (identifier))))) method: (identifier))))
'''
