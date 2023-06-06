
from flask import url_for,Flask,render_template,request
from PricePredicting import BostonLinearRegression
from flask_cors import cross_origin

app=Flask(__name__)

@cross_origin()
@app.route('/',methods=['GET','POST'])
def home():
    return render_template('index.html')
@cross_origin()
@app.route('/result',methods=['GET','POST'])
def result():
    if request.method == "POST":
        l=[]
        try:
            crim=float(request.form['crim'])
            zn = float(request.form['zn'])
            indus = float(request.form['indus'])
            chas =float (request.form['chas'])
            nox = float(request.form['nox'])
            rm = float(request.form['rm'])
            age =float (request.form['age'])
            dis = float(request.form['dis'])
            rad =float (request.form['rad'])
            tax = float(request.form['tax'])
            ptratio =float (request.form['ptratio'])
            b = float(request.form['b'])
            lstat = float(request.form['lstat'])
        except Exception as e:
            return render_template('error.html',error=str(e))
        try:
            record=[crim,zn,indus,chas,nox,rm,age,dis,rad,tax,ptratio,b,lstat]
        except Exception as e:
            return render_template('error.html',error=" in creating list")
        try:

            obj = BostonLinearRegression()
        except Exception as e:
            return render_template('error.html',error=str(e))
        try:

            obj.correlationmatrix()
            x=obj.df.drop(columns=['medv'])
            y=obj.df[['medv']]
            x_train, x_test, y_train, y_test = obj.split_df(x, y,test_size=0.3,random_state=345)
            obj.createModel()
            norm_df=obj.normalization(x_train)
            obj.trainModel()
            y_pred=obj.predict(x_test)
            obj.predict([record])
            final_ans=str(obj.output)
            final_ans=final_ans.strip("[]")

        except Exception as e:
            return render_template('error.html',error=str(e))


        return render_template('result.html',result=final_ans)
    else:
        return render_template("getmethod.html")

if __name__=='__main__':
    app.run(debug=True)




