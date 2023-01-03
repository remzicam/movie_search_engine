from pandas import read_pickle
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.colored_header import colored_header
from streamlit_option_menu import option_menu

max_seq_length = 256
repo_id = "all-MiniLM-L6-v2"
data_path = "detailed_movies_top_250_embeds.pkl.xz"
output_column_names = [
    "year",
    "duration",
    "genre",
    "stars",
    "summary",
    "poster_url",
    "trailer_url",
]
vertical_space = 2
st.set_page_config(layout="wide")

colored_header(
    label="SEARCH ENGINE&MOVIE RECOMMENDER: IMDB TOP 250 MOVIES",
    description="""Discover the best movies from the IMDB Top 250 list with advanced semantic search engine and movie recommender.
                    Simply enter a keyword, phrase, or even plot.
                    It provides you with a personalized selection of top-rated films!""",
    color_name="blue-70",
)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_data_model():
    """
    It loads the dataframe and the sentence embedding model.
    
    Returns:
      A tuple of the dataframe and the embedding model
    """
    
    df = read_pickle(data_path)
    embed_model = SentenceTransformer(repo_id)
    embed_model.max_seq_length = max_seq_length 
    return df, embed_model


def top_n_retriever(titles, similarity_scores, n, query_type):
    """
    It takes in a list of titles, a numpy array of similarity scores, the number of results to return,
    and the type of query (search engine or similar movies). It then returns the top n results
    
    Args:
      titles (list[str]): List of movie titles
      similarity_scores (ndarray): The cosine similarity scores of the query movie with all the movies
    in the dataset.
      n (int): The number of results to return
      query_type (str): This is the type of query. It can be either "Search Engine" or "Similar Movies".
    
    Returns:
      The top n movies that are similar to the query movie.
    """
    
    sim_scores = zip(titles, similarity_scores)
    sorted_sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    if query_type == "Search Engine":
        sorted_sim_scores = sorted_sim_scores[:n]

    if query_type == "Similar Movies":
        sorted_sim_scores = sorted_sim_scores[1 : n + 1]

    return [i[0] for i in sorted_sim_scores]


def grid_maker(movie_recs, df):
    """
    It takes the list of recommended movies and the dataframe as input and outputs a grid of movie
    posters and details
    
    Args:
      movie_recs (List[str]): - a list of movie titles
      df (object): the dataframe containing the movie data
    """

    for movie in movie_recs:
        poster_col, title_col = st.columns([1, 8])
        (year, duration, genre, stars, summary, poster_url, trailer_url) = (
            df[output_column_names][df.title == movie]
        ).values.flatten()
        poster_col.image(poster_url)
        poster_col.markdown(
            f'<a href={trailer_url}><button style="background-color:GreenYellow;">🎥Trailer</button></a>',
            unsafe_allow_html=True,
        )

        title_col.markdown(f"""<p>
                                <span style=color:#0068C9;font-style:bold;font-size:28px;>{movie} </span>
                                <span style=color:grey;font-style:italic;font-size:14px;> {year} | {duration} | {genre}</span>
                                <span style="background-color:rgba(0, 0, 0, 0.1);"><br>{stars}</span>
                                <span style="word-wrap:break-word;font-family:roboto;font-weight: 700;">
                                <br>{summary}</span>
                                </p>
                            """, unsafe_allow_html=True)
        add_vertical_space(vertical_space)


def filter_df(df, selected_page):
    """
    The function takes in a dataframe, and the selected page, and returns the selected movie, the
    filtered dataframe, and the top_n number of recommendations
    
    Args:
      df (object): the dataframe
      selected_page (str): the page that the user is on
    
    Returns:
      selected_movie, filtered_df, top_n
    """
    filtered_df = df.copy()
    text_input, genre_box, top_n_rec = st.columns([3, 1, 2])
    with genre_box:
        selected_genre = st.selectbox("Genre", genres_list)
    with top_n_rec:
        top_n = st.slider("Number of Recommendations", 1, 15, 5)

    if selected_genre != "All":
        filtered_df = df[df.genre.str.contains(selected_genre)]

    if selected_page == "Similar Movies":
        with text_input:
            selected_movie = st.selectbox("Movie", movie_list)
        return selected_movie, filtered_df, top_n

    if selected_page == "Search Engine":
        with text_input:
            query = st.text_input("Query", value="Mafia")
        return query, filtered_df, top_n


def get_results_button():
    """
    It creates a button that says "Get Results ◀" and returns it
    
    Returns:
      A button object.
    """
    _, _, col_center, _, _ = st.columns(5)
    return col_center.button("Get Results ◀")


df, embed_model = load_data_model()
df["trailer_url"] = df["trailer_url"].astype(str)
movie_list = df["title"].values
genres_list = list(set(df["genre"].str.split(", ").sum()))
genres_list.insert(0, "All")


selected_page = option_menu(
    menu_title=None,  # required
    options=["Search Engine", "Similar Movies"],  # required
    icons=["search", "film"],  # optional
    menu_icon="cast",  # optional
    default_index=0,  # optional
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"},
        "nav-link": {
            "font-size": "25px",
            "text-align": "left",
            "margin": "0px",
            "--hover-color": "#eee",
        },
        "nav-link-selected": {"background-color": "#0068C9"},
    },
)

if selected_page == "Search Engine":

    query, genre_df, top_n = filter_df(df, selected_page)
    query_embed = embed_model.encode(query)

    bt = get_results_button()

    if bt:
        if query == "":
            st.warning("You should type something", icon="⚠️")
        else:
            semantic_sims = [
                cosine_similarity([query_embed], [movie_embed]).item()
                for movie_embed in genre_df.embedding
            ]
            movie_recs = top_n_retriever(
                genre_df.title, semantic_sims, top_n, selected_page
            )
            add_vertical_space(vertical_space)
            grid_maker(movie_recs, genre_df)


if selected_page == "Similar Movies":
    st.info("Movies are recommended based on plot similarity!")
    selected_movie, genre_df, top_n = filter_df(df, selected_page)

    bt = get_results_button()
    if bt:
        movie_sims = [
            cosine_similarity(
                list(df.embedding[df.title == selected_movie]), [movie_embed]
            ).item()
            for movie_embed in genre_df.embedding
        ]
        movie_recs = top_n_retriever(genre_df.title, movie_sims, top_n, selected_page)
        add_vertical_space(vertical_space)
        grid_maker(movie_recs, genre_df)
