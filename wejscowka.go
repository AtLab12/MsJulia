COLOR := 'color'
BACKGROUNDCOLOR := 'background-color'
FONTSIZE := 'font-size'
HexElement := [a-zA-Z0-9]
HEX := '#' HexElement HexElement HexElement HexElement HexElement HexElement
STRING := [a-zA-Z]+
DIVELEMENT := 'div'
PELEMENT := 'p'

css := cssElement+
cssEleemnt := selector '{' bodyElementSC* bodyElementNOSC '}'

// selector element
div := DIVELEMENT class?
p := PELEMENT class?
class := '.' STRING class*

selector := complexSelector
selectorElement := (div | p | class | '*')
subSelector := '>' selectorElement
complexSelector := selectorElement subSelector*

// body element
bodyElementSC := (colorElement | fontElement) ';'
bodyElementNOSC := colorElement | fontElement
colorElement := (COLOR | BACKGROUNDCOLOR) ':' HEX
fontElement := FONTSIZE ':' [1-9][0-9]* 'px'