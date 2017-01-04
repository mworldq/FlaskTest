from flask_wtf import Form
from wtforms import TextField, BooleanField, StringField

class LoginForm(Form):
    im = StringField('im', default='')
    tx = StringField('tx', default='')
