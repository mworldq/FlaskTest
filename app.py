from flask import Flask, request
from flask import render_template
from forms import LoginForm
import base64
import predict

app = Flask(__name__)

app.config['SECRET_KEY']='xxx'

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def hello():
    # file = request.files['i']
    # if(file)
    #     print "file", file
    form = LoginForm()
    imgData = form.im.data.replace('data:','')
    print 'imgData', imgData
    result = None
    if imgData:
        imgData = base64.b64decode(imgData)
        leniyimg = open('data.png', 'wb')
        leniyimg.write(imgData)
        leniyimg.close()

        imvalue = predict.imageprepare('./data.png')
        predint = predict.predictint(imvalue)
        print (predint[0])  # first value in list
        result = predint[0]

    return render_template('predict.html', result=result)


    # im = misc.imread("./data.png")
    # img = im.reshape((1,784))
    # clf = joblib.load('model/ok.m')
    # l = clf.predict(img)
    # print 'predict: %s ' % (l[0])
    # return render_template('Hello.html', result=l[0])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)