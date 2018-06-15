
:trollface: Hater News
==========

Haterz Gonna Hate. But now you know who the haterz are.

:bulb: The Idea.
---------
Using [Scikit Learn](http://scikit-learn.org/stable/)'s [CountVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html), some additional [hand-rolled features](https://github.com/kevinmcalear/hater_news/blob/master/its_pkl_time.py), and [Logistic Regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) you can start to build out a way to analyize all the comments a user ever has posted in [Hacker News](https://news.ycombinator.com/) or [Reddit](http://www.reddit.com/) and rank how big of a troll they are.


:squirrel: How It's Built.
---------

Feel free to download either [the HTML](https://github.com/kevinmcalear/hater_news/blob/master/haterz_classification.html) or [iPython Notebook](http://nbviewer.ipython.org/github/kevinmcalear/hater_news/blob/master/haterz_classification.ipynb) files from this repo. I tried to detail out my explorations and assumtions about the model so you can follow my logic and learn why and how I built everything.

***Side Note:*** I'm going to make it a [Chrome App](https://developer.chrome.com/extensions/getstarted) soon. I'm also going to build versions of this for Twitter, reddit, Instagram, Facebook, and maybe even dating apps. If you want to help, reach out! :)


:computer: Run The App Yourself.
---------

#### Preparing the app

Execute the following commands to clone this repo:

	$ git clone git@github.com:kevinmcalear/hater_news.git
	$ cd hater_news

You now have a functioning git repository that contains the app as well as a [requirements.txt](https://github.com/kevinmcalear/hater_news/blob/master/requirements.txt) and a [Procfile](https://github.com/kevinmcalear/hater_news/blob/master/Procfile), which are required to run our app in Heroku or using foreman.

#### Running the app locally

You can run the app locally one of two ways:

By using foreman:

	$ foreman start

Or by running the app.py file with python:

	$ python app.py

[foreman](https://github.com/ddollar/foreman) gives you a decent preview of how the app will run when on Heroku. I have printed out several things to the console for debugging and exploration which will show up if you run the app with the `python app.py` command.

#### :globe_with_meridians: Hosting with Heroku

[Sign Up for Heroku](https://id.heroku.com/signup)

[Install Heroku Toolbelt](https://toolbelt.heroku.com/osx)

Once installed, you can use the heroku command from your command shell.

**Log in using the email address and password you used when creating your Heroku account:**

	$ heroku login
	Enter your Heroku credentials.
	Email: python@example.com
	Password:
	Could not find an existing public key.
	Would you like to generate one? [Yn]
	Generating new SSH public key.
	Uploading ssh public key /Users/username/.ssh/id_rsa.pub

Press enter at the prompt to upload your existing ssh key or create a new one, used for pushing code later on.

To check that your key was added, type heroku keys. If your key isn’t there, you can add it manually by typing heroku keys:add. For more information about SSH keys, see [Managing Your SSH Keys](https://devcenter.heroku.com/articles/keys).


#### Deploy the app to Heroku

Create an app on Heroku, which prepares Heroku to receive your source code. * **Note:** we need to use [this buildpack](https://github.com/thenovices/heroku-buildpack-scipy) to get everything to work for sklearn and scipy.

For a new app:

	heroku create --buildpack https://github.com/thenovices/heroku-buildpack-scipy

For an existing app:

	heroku config:set BUILDPACK_URL=https://github.com/thenovices/heroku-buildpack-scipy

This also creates a remote repository (called heroku) which it configures in your local git repo. Heroku generates a random name for your app.  * **Note:**  you can pass a parameter to specify your own name, or rename it later with `heroku apps:rename`.

Now deploy your code:

	$ git push heroku master
	git push heroku master
	Fetching repository, done.
	Counting objects: 7, done.
	Delta compression using up to 8 threads.
	Compressing objects: 100% (4/4), done.
	Writing objects: 100% (4/4), 457 bytes | 0 bytes/s, done.
	Total 4 (delta 2), reused 0 (delta 0)

	-----> Fetching custom git buildpack... done
	-----> Python app detected
	-----> No runtime.txt provided; assuming python-2.7.4.
	-----> Using Python runtime (python-2.7.4)
	-----> Detected numpy/scipy in requirements.txt. Downloading prebuilt binaries.
	-----> Using cached binaries.
	-----> Existing NumPy (1.8.1) package detected.
	-----> Existing SciPy (0.14.0) package detected.
	-----> Installing dependencies using Pip (1.3.1)
	       Cleaning up...

	-----> Discovering process types
       Procfile declares types -> web

	-----> Compressing... done, 95.5MB
	-----> Launching... done, v39
	       https://haternews.herokuapp.com/ deployed to Heroku

	To git@heroku.com:haternews.git
	   f335692..bfb7017  master -> master

The application is now deployed. Ensure that at least one instance of the app is running:

	$ heroku ps:scale web=1

Now visit the app at the URL generated by its app name. As a handy shortcut, you can open the website as follows:

	$ heroku open

:metal: Acknowledgments
---------------
#### The Data Science
Shout Outs to [@gavinmh](https://github.com/gavinmh), [@jamesbev](https://github.com/jamesbev), [@ShawnOakley](https://github.com/ShawnOakley), and [General Assembly](https://generalassemb.ly/)'s [Data Science Program](https://generalassemb.ly/education/data-science). Without the help I've had I would have no idea how to do anything Data Science related.

#### Everything Else
* My [Training Data](https://www.kaggle.com/c/detecting-insults-in-social-commentary/data).
* The [Hacker News API](https://github.com/HackerNews/API).
* [PRAW ("Python Reddit API Wrapper")](https://github.com/praw-dev/praw) for Reddit's API.
* Twitter API [tweepy](https://github.com/tweepy/tweepy) python wrapper.
* [Flask](http://flask.pocoo.org/) web framework.
* [Jinja2](http://jinja.pocoo.org/docs/dev/) templating.
* [JQuery](http://jquery.com/) is life.
* [Snap.svg](http://snapsvg.io/) SVG javascript library.
* [SweetAlert.js](https://github.com/t4t5/sweetalert) plugin for alerts.
* Guts of [Loading Screen Tutorial](http://tympanus.net/codrops/2014/04/23/page-loading-effects/).
* [Custom Heroku buildpack](https://github.com/thenovices/heroku-buildpack-scipy) for Python with Numpy 1.8.1 and SciPy 0.14.0.
* [Heroku-config](https://github.com/ddollar/heroku-config) for adding env keys.
