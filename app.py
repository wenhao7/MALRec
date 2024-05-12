from flask import Flask, request, jsonify, render_template
from flask import send_from_directory
#from app import preprocessing, recommendation, scrape_anime_user_info
from app.preprocessing import *
from app.recommendation import *
from app.scrape_anime_user_info import *

app = Flask(__name__)

# File Paths
anime_info_file = './data/seasonal_anime.csv'
user_ratings_file = './data/cleaned_user_ratings.csv'
user_input_file = './data/user_input.csv'
recommendations_file = './data/predictions.csv'

# Initialise variables
usernames = []
topn=10
cb_weight = 0.5

# Import dataset
df_content = import_seasonal(anime_info_file)
df_ratings = pd.read_csv(user_ratings_file)

# Initialise recommender system object
recommender = HybridRecommender(ContentBasedRecommender,
                                CollaborativeRecommender,
                                df_content,
                                df_ratings,
                                cb_weight=cb_weight)

###############################################

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST', 'GET'])
def recommend():
    st = datetime.now()
    if request.method == 'POST':
        global usernames
        usernames = [str(x) for x in request.form.getlist('username')]
        usernames_string = ', '.join(usernames)
        print(usernames)
        scrape_user_animelist([str(x) for x in request.form.values()], req_head, output_file=user_input_file)
        user_input = pd.read_csv(user_input_file, delimiter='|')
        print(f'Time Taken to scrape user data : {datetime.now() - st}')
        # predictions = recommender.cb_model.df_content.sort_values('Score', ascending=False)
        # predictions.to_csv(recommendations_file, sep='|', index=False)
        predictions = recommender.predict(user_input).reset_index(drop=True)
        predictions = predictions.merge(recommender.cb_model.df_content, how='left', left_on='MAL_Id', right_on='MAL_Id')
    elif request.method == 'GET':
        predictions = import_seasonal(recommendations_file)
        usernames_string = ', '.join(usernames)
    # return render_template('recommend.html', recommendations=predictions[:topn], usernames_string=usernames_string)
    
    print(f'Time Taken to predict : {datetime.now() - st}')
    predictions.to_csv(recommendations_file, sep='|', index=False)
    return render_template('recommend.html', recommendations=predictions[:topn], usernames_string=usernames_string)

@app.route('/filter', methods=['POST'])
def filter_rec():
    predictions = import_seasonal(recommendations_file)
    predictions.Genres = [', '.join(map(str,l)) for l in predictions.Genres]
    usernames_string = ', '.join(usernames)
    filters = [str(x) for x in request.form.getlist('genre')]
    print(filters)
    return render_template('filtered_rec.html', recommendations=predictions, usernames_string=usernames_string, filters=filters)

if __name__ == '__main__':
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(host='0.0.0.0', port=5000)