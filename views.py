from flask import Blueprint, render_template

views = Blueprint(__name__, "views")

@views.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__": 
  app.run(debug=True)
  app.debug = True