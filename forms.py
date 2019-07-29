from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField


class UploadForm(FlaskForm):

    validators = [
        FileRequired(message='There was no file!'),
        FileAllowed(['txt'], message='Must be a txt file!')
    ]

    input_file = FileField('', validators=validators)
    submit = SubmitField(label="Upload")