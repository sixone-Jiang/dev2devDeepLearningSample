from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

import os
from PIL import Image

from model.predict import get_Model, predictImg

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'upload/'

modelPath = './model/model2.h5'


# 加载模型
mymodel = get_Model(modelPath)

'''
@app.route('/upload')
def upload_file():
    return render_template('upload.html')
'''

@app.route('/uploader',methods=['GET','POST'])
def uploader():
    if request.method == 'POST':
        f = request.files['file']
        #print(request.files)
        #print(type(f))
        # TODO list
        # 1. fileName need hash
        # 2. uploadData need check Image is, 
        # 3. uploadData need clear before run 
        savePath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
        f.save(savePath)
        ans = predictImg(mymodel, savePath)

        # return 'file uploaded successfully'
        return ans

    else:
        #return render_template('upload.html')
        return 'Error'

if __name__ == '__main__':
   app.run('0.0.0.0', '13406')