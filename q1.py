import streamlit as st
import pandas as pd
import numpy as np
import sympy as sp
from sympy.logic.boolalg import truth_table
import schemdraw
from schemdraw import logic
from schemdraw.parsing import logicparse
import re
from jinja2 import Template

hide_menu = """
<style>
#MainMenu {
    visibility: hidden;
}
footer {
    visibility: visible;
}
footer:after{
    content:"By: Oswaldo Cadenas, 2022";
    display: block;
    position: relative;
    color: tomato;
}
</style>
"""

# Useful template to generate SystemVerilog code 
# requires the input and output labels
# and sympy equations for all outputs
template = """
// replace 'change_me' with a suitable name
module change_me(input logic {% for lit in inputs %}{{lit}}, {% endfor %}
                 output logic {% for lit in outputs[:-1] %}{{lit}}, {% endfor %}{{outputs[-1]}});
// use blocking assignment "=" in combinational blocks 
{% if outputs|length == 1 -%}  
always_comb {{outputs[0]}} = {{eqs[outputs[0]]}};

{% else -%}
always_comb begin
{% for key, value in eqs.items() -%}
{{key|indent(2, True)}} = {{value}}; 
{% endfor -%}
end

{% endif -%}

endmodule
"""

# A function to generate Karnaugh maps groups as expected my schemdraw
# see: https://schemdraw.readthedocs.io/en/latest/elements/logic.html#karnaugh-maps
def sopm2groups(minterm, labels):
    table = truth_table(minterm, labels)
    algo = list(table)
    summary = [t[0] for t in algo if t[1] == True]
    r = len(summary)
    summary = np.array(summary).reshape(r, len(labels))
    lstr = []
    for i, sym in enumerate(labels):
        p = summary[:,i]
        s = np.count_nonzero(p)
        # print(i, p, s)
        if s == r:
            lstr.append('1')
        if s == 0:
            lstr.append('0')
        if s > 0 and s < r:
            lstr.append('.')
            
    return ''.join(lstr)

# some colors to apply to groups in Karnough maps
colors = [{'color': 'black', 'fill': '#00000033'}, 
          {'color': 'purple', 'fill': '#bf00ff33'},
           {'color': 'green', 'fill': '#00ff0033'},
        {'color': 'pink', 'fill': '#ff00ff33'},
         {'color': 'yellow', 'fill': '#ffff0033'},
        {'color': 'blue', 'fill': '#0000ff33'}, 
        {'color': 'red', 'fill': '#ff000033'},
        ]

# converts sympy logic equations to Latex
def totex(expr):
    pnot = re.compile(r'~+')
    por = re.compile(r'\|+')
    pand = re.compile(r'&+')
    expr1 = pnot.sub(r'\\overline ', expr)
    expr1 = por.sub(r'+', expr1)
    expr2 = pand.sub(r'\\cdot ', expr1)
    return expr2

# truth table format for schemdraw
# see: https://schemdraw.readthedocs.io/en/latest/elements/logic.html#karnaugh-maps
def ttodraw(names, minterms, dontcares):
    isize = len(names)
    rtab = []
    for i in range(2**isize):
        a = f'{bin(i)[2:]:>0{isize}s}'
        if i in minterms:
            b = '1'
            rtab.append((a,b))
        if i in dontcares:
            b = 'x'
            rtab.append((a,b))
    return rtab

# A generic class for a combination problem from a truth table
class TruthTable:
    def __init__(self, inputs, outputs):
        if len(inputs) == 0:
            print ('Error: number of inputs has to be at least 1')
            raise Exception()
        else:
            self.inames = inputs
            self.isymbols = sp.symbols(inputs)
            self.i = len(inputs)
        if len(outputs) == 0:
            print ('Error: number of outputs has to be at least 1')
            raise Exception()
        else:
            self.onames = outputs
            self.osymbols = sp.symbols(outputs)
            self.o = len(outputs)
        # equations
        self.oeqs = {}
        self.minterms = {}
        self.dontcares = {}
        for o in outputs:
            self.oeqs[o] = None
            self.dontcares[o] = []
        frt = '#0' + str(2+self.i) + 'b'
        table = [list(format(i, frt)[2:]) for i in range(2**self.i)]
        self.tt = pd.DataFrame(np.array(table), columns=self.inames)
        for i in range(self.o):
            self.tt[self.onames[i]] = np.array([0]*2**self.i, dtype=str)

    # returns a table as a Pandas DataFrame
    def get_tt(self):
        return self.tt

    # minterms for a truth table output as a list
    def set_minterms(self, minterms, name):
        self.minterms[name] = minterms
        if name in self.onames:
            for pos in minterms:
                if (pos >= 0) and (pos < 2**self.i):
                    self.tt.at[pos, name] = str(1)
                else:
                    print ('Minterm {} does not exist'.format(pos))
        else:
            print ("Output name '{}' is not in table".format(name))
            raise Exception()

    # don't cares for a truth table output as a list
    def set_dontcares(self, dontcares, name):
        if len(dontcares) != 0:
            self.dontcares[name] = dontcares
            if name in self.onames:
                for pos in dontcares:
                    if (pos >= 0) and (pos < 2**self.i):
                        self.tt.at[pos, name] = 'x'
                    else:
                        print ('Minterm {} does not exist'.format(pos))
            else:
                print ("Output name '{}' is not in table".format(name))
                raise Exception()
        else:
            self.dontcares[name] = []

    # returns a list
    def get_minterms(self, name):
        if name in self.onames:
            return self.minterms[name]
        else:
            print ("Output name '{}' is not in table".format(name))
            raise Exception()

    # returns a list
    def get_dontcares(self, name):
        if name in self.onames:
            return self.dontcares[name]
        else:
            print ("Output name '{}' is not in table".format(name))
            raise Exception()

    # a don't care must not be at same position of a minterm
    def check_dontcares(self, name):
        ms = set(self.minterms[name])
        ds = set(self.dontcares[name])
        return ms.intersection(ds)

    # sympy logic expression for truth table output
    # this builds a sum-of-minterms for truth table output 'name'
    def SOP(self, name):
        minterms = self.get_minterms(name)
        lstr = ''
        for v in self.inames:
            lstr += v 
            lstr += ' '
        lstr = lstr.strip()
        expr = ''
        for m in minterms[0:-1]:      
            expr += repr(sp.SOPform(sp.symbols(lstr), [m]))
            expr += ' + '
        expr += repr(sp.SOPform(sp.symbols(lstr), [minterms[-1]]))
        return expr

    # sum-of-minterms simpy logic expression for truth table output 'name'
    def mSOP(self, name):
        if name in self.onames:
            minterms = self.get_minterms(name)
            dontcares = self.get_dontcares(name)
            lstr = ''
            for v in self.inames:
                lstr += v 
                lstr += ' '
            lstr = lstr.strip()
            expr = repr(sp.SOPform(sp.symbols(lstr), minterms, dontcares))
            # texexpr = name + '=' + totex(expr) 
        else:
            print ("Output name '{}' is not in table".format(name))
            raise Exception()
        self.oeqs[name] = expr
        return expr

    # Save the Karnaugh map for truth table output 'output'
    # as an image using schemdraw
    def kmap(self, idx, output):
        # A, B, C = sp.symbols('A B C')
        # ltab_o = truth_table(self.oeqs[output], self.isymbols)
        rtab = ttodraw(self.inames, self.get_minterms(output), self.get_dontcares(output))
        lgroups = self.get_groups(output)
        with schemdraw.Drawing(backend='matplotlib', show=False) as d:
            d.add(logic.Kmap(names=''.join(self.inames), truthtable=rtab, groups=lgroups))
            img_name = f'mymap{idx}.png'
            d.save(fname=img_name, dpi= 300)  

    # collects all Karnaugh map groups for truth table output
    def get_groups(self, output):
        lsymbols = self.isymbols
        temp = self.oeqs[output].split('|')
        temp = [lit.strip() for lit in temp]
        groups = []
        for minterm in temp:
            groups.append(sopm2groups(minterm, lsymbols))
        pgroups = {}
        start = np.random.randint(0, len(colors))
        for i, g in enumerate(groups):
            pgroups[g] = colors[(start + i) % len(colors)]
        return pgroups

    # saves as an image circuit for truth table output 
    # derived from the minimised sop
    def circuit(self, idx, output):
        lexpr = self.oeqs[output]
        logic = logicparse(lexpr, outlabel=output) 
        with schemdraw.Drawing(backend='svg', show=False) as d:
            d = logic
            img_name = f'mycircuit{idx}.png'
            d.save(fname=img_name, dpi= 300)


st.sidebar.image('LSBU_2020_BO.png', width=140)
# st.title('EEE_4_DLD: Combinational Logic')
title = '<p style="font-family:verdana; color:#ff1010; font-size: 36px;"><b>EEE_4_DLD: Combinational Logic</b></p>'
st.markdown(hide_menu, unsafe_allow_html=True)
st.markdown(title, unsafe_allow_html=True)

st.sidebar.write('Truth table variables')

def clear_form():
    st.session_state["input"] = ""
    st.session_state["output"] = ""

reset = st.sidebar.button('Reset', on_click=clear_form)

if "submit_boolean" not in st.session_state:
    st.session_state.submit_boolean = False

if "msop_boolean" not in st.session_state:
    st.session_state.msop_boolean = False

# inputs
input_labels = st.sidebar.text_input('Input labels such as: A, B, Ci', key='input', placeholder="Separate multiple labels with commas")
inputs = input_labels.split(',')
inputs = [ch.strip() for ch in inputs]
n = len(inputs)


outputs = ['']
m = 1
if (n >= 1) and (n <= 4):
    # Outputs
    if (n > 0) and (inputs[0] != ''):
        output_labels = st.sidebar.text_input('Outputs labels such as Co, S', key = 'output', placeholder="Separate multiple labels with commas")
        outputs = output_labels.split(',')
        outputs = [ch.strip() for ch in outputs]
        m = len(outputs)
        if m > 2:
            st.sidebar.warning(f"Keep number of outputs to 2 as maximum")
            st.stop()
else:
    st.sidebar.info('Keep number of inputs between 1 and 4')

if reset:
    n = 0
    m = 0
    st.session_state.msop_boolean = False
    st.session_state.submit_boolean = False


if (m >= 1) and (outputs[0] != ''):
    mytable = TruthTable(inputs, outputs)

# outputs = st.sidebar.selectbox('Outputs', [1, 2])
if (m >= 1) and (outputs[0] != ''):
    for ol in outputs:
        minterms_name = 'minterms_' + ol
        message = 'Minterms of ' + ol
        minterms_name = st.sidebar.multiselect(message, range(2**len(inputs)))
        mytable.set_minterms(minterms_name, ol)
    dontcares = st.sidebar.checkbox("Don't Cares")

if (m >= 1) and (outputs[0] != ''):
    if dontcares:
        for ol in outputs:
            dontcares_name = 'dontcares_' + ol
            message = "Don't cares of " + ol
            dontcares_name = st.sidebar.multiselect(message, range(2**len(inputs)))
            mytable.set_dontcares(dontcares_name, ol)
            common = list(mytable.check_dontcares(ol))
            if len(common) > 0:
                st.sidebar.warning(f"Row {common[0]} must be EITHER a minterm OR don'tcare")
                st.stop()
    submit = st.sidebar.button('Submit')
    # msop = st.sidebar.button('Minimised SOP')


if (outputs[0] != ''):
    tab1, tab2, tab3, tab4 = st.tabs(["Table", "Maps", "Circuit", "SystemVerilog"])

# col1, col2 = st.columns([2, 3])
if (m >= 1) and (outputs[0] != ''):
    if submit:
        st.session_state.submit_boolean = True



if (m >= 1) and (outputs[0] != ''):
    with tab1:
        if st.session_state.submit_boolean:
            st.write('Truth table')
            st.dataframe(mytable.get_tt())
            st.write('Sum-of-Products of minterms')
            for ol in mytable.onames:
                leq = ol + ' = ' + totex(mytable.SOP(ol))
                st.latex(leq)

if (m >= 1) and (outputs[0] != ''):
    with tab2:
        msop = st.button('Minimised SOP')
        col0, col1 = st.columns([4, 4])
        if msop:
            st.session_state.msop_boolean = True
        if st.session_state.msop_boolean:
            with col0:
                ol = mytable.onames[0]
                message = 'Minimised SOP Boolean equation for ' + ol
                st.write(message)
                mytable.mSOP(ol)
                mytable.kmap(0, ol)
                img_name = 'mymap' + '0' + '.png'
                st.image(img_name, channels="RGB")
                meq0 = ol + ' = ' + totex(mytable.oeqs[ol])
                # st.latex(meq0)
            with col1:
                if m > 1:
                    ol = mytable.onames[1]
                    message = 'Minimised SOP Boolean equation for ' + ol
                    st.write(message)
                    mytable.mSOP(ol)
                    mytable.kmap(1, ol)
                    img_name = 'mymap' + '1' + '.png'
                    st.image(img_name, channels="RGB")
                    meq1 = ol + ' = ' + totex(mytable.oeqs[ol])
                    # st.latex(meq1)
            # display latex equations
            st.latex(meq0)
            if m > 1:
                st.latex(meq1)

           
if (m >= 1) and (outputs[0] != ''):
    with tab3:
        col0, col1 = st.columns([4, 4])
        if st.session_state.msop_boolean:
            with col0:
                ol = mytable.onames[0]
                mytable.circuit(0, ol)
                img_name = 'mycircuit' + '0' + '.png'
                message = 'Boolean logic diagram for ' + ol
                st.write(message)
                st.image(img_name, channels="RGB")
                meq = ol + ' = ' + totex(mytable.oeqs[ol])
                st.latex(meq)
            with col1:
                if m > 1:
                    ol = mytable.onames[1]
                    mytable.circuit(1, ol)
                    img_name = 'mycircuit' + '1' + '.png'
                    message = 'Boolean logic diagram for ' + ol
                    st.write(message)
                    st.image(img_name, channels="RGB")
                    meq = ol + ' = ' + totex(mytable.oeqs[ol])
                    st.latex(meq)
                  
           
if (m >= 1) and (outputs[0] != ''):
    with tab4:
        if st.session_state.msop_boolean:
            leqs = {}
            for ol in mytable.onames:
                lexpr = mytable.oeqs[ol]
                leqs[ol] = lexpr
            my_template = Template(template)
            mycode = my_template.render(inputs = mytable.inames, outputs=mytable.onames, eqs = leqs)
            st.code(mycode, language='verilog')