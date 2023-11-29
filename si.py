from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

data = pd.DataFrame(columns=['ID', 'Choice1', 'Choice2', 'Choice3', 'Choice4', 'Choice5'])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        id = request.form.get('id')
        choice1 = request.form.get('choice1')
        choice2 = request.form.get('choice2')
        choice3 = request.form.get('choice3')
        choice4 = request.form.get('choice4')
        choice5 = request.form.get('choice5')

        global data
        data = data.append({'ID': id, 'Choice1': choice1, 'Choice2': choice2, 'Choice3': choice3, 'Choice4': choice4, 'Choice5': choice5}, ignore_index=True)

        analysis = data.describe()

        return render_template('index.html', analysis=analysis.to_html())

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
