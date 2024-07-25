import pandas as pd
import texthero as hero

# Texthero: Text preprocessing, representation and visualization from zero to hero.
# Combines NLTK, SpaCy, Gensim, TextBlob, Sklearn (pip install texthero)

print(help(hero))

"""Text Preprocessing"""
text = "It's a pleasant   day at Bangaloré; at / (10:30) am"
series = pd.Series(text)
print(series)

print('Remove Digits\n', hero.remove_digits(series))
print('Remove Punctuations\n', hero.remove_punctuation(series))
print('Remove Brackets\n', hero.remove_brackets(series))
print('Remove Diacritics\n', hero.remove_diacritics(series))
print('Remove Whitespace\n', hero.remove_whitespace(series))
print('Remove Stopwords\n', hero.remove_stopwords(series))
print(hero.clean(series))

df = pd.read_csv("https://github.com/jbesomi/texthero/raw/master/dataset/bbcsport.csv")
print(df.head())

df['pca'] = (df['text']
             .pipe(hero.clean)
             .pipe(hero.tfidf)  # vectorizing
             .pipe(hero.pca))
hero.scatterplot(df, 'pca', color='topic', title='PCA BBC Sport news')
print(df.head())

# df = pd.read_csv("https://github.com/jbesomi/texthero/raw/master/dataset/bbcsport.csv")
# print(df.head())
#
# df['tfidf'] = (df['text']
#                .pipe(hero.clean)
#                .pipe(hero.tfidf))
#
# # Kmeans
# df['kmeans_labels'] = (df['tfidf']
#                        .pipe(hero.KMeans, n_clusters=5)
#                        .astype(str))
#
# df['pca'] = df['tfidf'].pipe(hero.pca)
#
# hero.scatterplot(df, 'pca', color='kmeans_labels', title='K-means BBC Sport news')
# print(df.head())
