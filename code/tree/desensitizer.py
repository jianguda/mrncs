import re
import string

from tree import shared

go = ['const', 'var', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'byte', 'rune',
      'int', 'uint', 'uintptr', 'float32', 'float64', 'complex64', 'complex128', 'bool', 'string',
      'go', 'select', 'chan', 'defer', 'error']
java = ['char', 'boolean', 'byte', 'short', 'int', 'long', 'float', 'double', 'throws', 'package', 'native',
        'strictfp', 'synchronized', 'transient', 'volatile', 'this', 'extends', 'abstract', 'final',
        'protected', 'public', 'private', 'static']
javascript = ['const', 'var', 'let', 'byte', 'arguments', 'debugger', 'catch', 'enum', 'delete', 'eval', 'void',
              'transient', 'synchronized', 'volatile', 'async', 'await', 'require']
php = ['const', 'string', 'int', 'float', 'bool', 'extends', 'implements', 'insteadof', 'abstract', 'include_once',
       'new', 'clone', 'echo', 'use', 'namespace', 'endwhile', 'endforeach', 'enddeclare', 'endswitch', 'endfor']
python = ['int', 'double', 'str', 'del', 'print', 'global', 'nonlocal', 'import', 'from', 'alias', 'as', 'with',
          'async', 'await', 'callable', 'lambda', 'self', 'super', 'return', 'yield', 'exit']
ruby = ['begin', 'end', 'do', 'undef', 'defined', 'callable']
others = ['try', 'catch', 'finally', 'ensure', 'rescue', 'retry', 'except', 'raise', 'throw', 'assert',
          'nil', 'none', 'null', 'nan', 'void']
noise_tokens = set(go + java + javascript + php + python + ruby + others)

case_tokens = {'if', 'then', 'else', 'elif', 'elsif', 'elseif', 'unless',
               'switch', 'case', 'default', 'fallthrough', 'declare'}
loop_tokens = {'for', 'while', 'do', 'redo', 'when', 'until', 'foreach'}
goto_tokens = {'goto', 'break', 'continue', 'next', 'pass'}
define_tokens = {'function', 'func', 'def', 'module', 'class', 'interface', 'struct', 'trait'}
assign_tokens = {'=', ':='}
judge_tokens = {'==', '<', '>', '!=', '<=', '>=', '===', '<>', '<=>', '!', '&', '|', '&&', '||',
                'and', 'or', 'not', 'xor', 'is', 'in', 'instanceof', 'typeof'}
operate_tokens = {'+', '-', '*', '/', '%', '+=', '-=', '*=', '/=', '%=', '**', '//', '=~'}
invoke_tokens = {'(', ')'}
access_tokens = {'.', '::', '[', ']', '{', '}'}
literal_tokens = {'\'', '"'}


def desensitize(token):
    token = str(token).lower().strip()
    if not shared.DESENSITIZE:
        return formalize(token)
    if token.isdigit():
        return 'number'
    elif token.isspace():
        return 'blank'
    # why we consider "if keyword in token" instead of "if token in keywords"?
    # because of the compatibility over both real tokens and sentences (from TS)
    tokens = token.split()
    tokens = set(filter(None, tokens))
    tokens = set(filter(lambda x: x not in noise_tokens, tokens))
    if case_tokens & tokens:
        return 'case'
    elif loop_tokens & tokens:
        return 'loop'
    elif goto_tokens & tokens:
        return 'goto'
    elif define_tokens & tokens:
        return 'module'
    elif assign_tokens & tokens:
        return 'assign'
    elif judge_tokens & tokens:
        return 'judge'
    elif operate_tokens & tokens:
        return 'operate'
    else:
        tokens = set(token)
        if invoke_tokens & tokens:
            return 'invoke'
        elif access_tokens & tokens:
            return 'access'
        elif literal_tokens & tokens:
            return 'literal'
        else:
            return formalize(token)


def formalize(token):
    # for code tokens
    token = str(token).lower().strip()
    token = re.sub(f'[{string.punctuation}0-9]', '', token)
    if not token:
        token = '@'
    return token


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
