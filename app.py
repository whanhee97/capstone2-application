from tomotopy import coherence as ch
import tomotopy as tp
from datetime import datetime
import re
from flask import Flask, render_template, request, session
import pandas as pd
from pandas import DataFrame as df
import numpy as np
import random
from collections import Counter
from konlpy.tag import Okt
okt = Okt()

app = Flask(__name__)
app.secret_key = 'aaaaavvvvbbbb'

review_file = {"경희대 국제캠퍼스": "reviews_youngtong.xlsx",
               "경기도 의왕시 청계동": "reviews_uiwang.xlsx"}


def topic_modeling(searched_reviews):
    # 데이터 프레임의 '리뷰' 열의 값들을 str 형식으로 바꾸기
    searched_reviews.리뷰 = searched_reviews.리뷰.astype(str)

    # 중복데이터 삭제
    searched_reviews.drop_duplicates(subset=['리뷰'], inplace=True)

    # 한글이 아니면 빈 문자열로 바꾸기
    searched_reviews['리뷰'] = searched_reviews['리뷰'].str.replace(
        '[^ㄱ-ㅎㅏ-ㅣ가-힣]', ' ', regex=True)

    # 빈 문자열 NAN 으로 바꾸기
    searched_reviews = searched_reviews.replace({'': np.nan})
    searched_reviews = searched_reviews.replace(
        r'^\s*$', np.nan, regex=True)

    # NAN 이 있는 행 삭제
    searched_reviews.dropna(how='any', inplace=True)

    # 인덱스 차곡차곡
    searched_reviews = searched_reviews.reset_index(drop=True)

    # 리뷰 데이터를 리스트로 변환
    Data_list = searched_reviews.리뷰.values.tolist()
    # 정규화 처리
    Data_list = list(map(okt.normalize, Data_list))

    # 명사,형용사 추출
    data_word = []

    # 한글자 짜리는 모두 없애야 함
    stopword = ['같다', '이다', '있다', '여기', '항상', '완전',
                '정말', '너무', '보고', '오늘', '역시', '이번', '다음', '아주']

    for i, document in enumerate(Data_list):
        clean_words = []
        for word in okt.pos(document, stem=True):  # 어간 추출
            if word[1] in ['Noun', 'Adjective']:
                if len(word[0]) >= 2 and word[0] not in stopword:
                    clean_words.append(word[0])
        data_word.append(clean_words)

    def compute_coherence_values(data_word, limit, start=4, step=2):

        coherence_values = []
        model_list = []
        for num_topics in range(start, limit, step):
            model = tp.LDAModel(
                k=num_topics, alpha=0.1, eta=0.01, min_cf=20, tw=tp.TermWeight.PMI, rm_top=1)
            for i, line in enumerate(data_word):
                if not line:
                    line.append(" ")
                model.add_doc(line)
            model.train(1500)  # 학습 정도
            model_list.append(model)
            coherence_model = ch.Coherence(
                model, coherence='c_v', window_size=0, targets=None, top_n=10, eps=1e-12, gamma=1.0)
            coherence_values.append(coherence_model.get_score())
        return model_list, coherence_values

    # 토픽 수를 14~ 25 로 해서 각각의 모델과 일관성 점수를 계산한 리스트 얻기
    model_list, coherence_values = compute_coherence_values(
        data_word=data_word, start=14, limit=25, step=2)
    # 최적의 모델 얻기
    limit = 25
    start = 14
    step = 2
    x = range(start, limit, step)
    topic_num = 0
    count = 0
    max_coherence = 0
    model_list_num = 0
    for m, cv in zip(x, coherence_values):
        coherence = cv
        if coherence >= max_coherence:
            max_coherence = coherence
            topic_num = m
            model_list_num = count
        count = count+1

    # 최적의 모델
    optimal_model = model_list[model_list_num]
    # 주제 단어 후보군 뽑기
    extractor = tp.label.PMIExtractor(
        min_cf=15, min_df=5, min_len=2, max_len=20, max_cand=10000)
    cands = extractor.extract(optimal_model)

    labels_per_topic = []

    labeler = tp.label.FoRelevance(
        optimal_model, cands, min_df=5, smoothing=0.01, mu=0.25)

    topics = []
    for k in range(optimal_model.k):
        labels_per_topic.append(labeler.get_topic_labels(k, top_n=2)[
                                0][0]+' + '+labeler.get_topic_labels(k, top_n=2)[1][0])
        keywords = []
        for word, prob in optimal_model.get_topic_words(k, top_n=10):
            keywords.append([word, prob])
        topics.append([k, labeler.get_topic_labels(k, top_n=2)[
            0][0]+' + '+labeler.get_topic_labels(k, top_n=2)[1][0], keywords])

    def format_topics_sentences(ldamodel=optimal_model):
        new_doc = ldamodel.docs
        sent_topics_df = pd.DataFrame()

        for i in range(len(data_word)):
            topics = new_doc[i].get_topics()
            for j, (topic_num, prop_topic) in enumerate(topics):
                if j == 0:
                    wp = optimal_model.get_topic_words(
                        topic_num, top_n=10)
                    topic_keywords = ", ".join(
                        [word for word, prop in wp])
                    topic_label = labels_per_topic[topics[j][0]]
                    sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(
                        prop_topic, 4), topic_label, topic_keywords]), ignore_index=True)
                else:
                    break
        sent_topics_df.columns = [
            'Dominant_Topic', 'Perc_Contribution', 'Topic_Label', 'Topic_Keywords']
        sent_topics_df = pd.concat([sent_topics_df, searched_reviews['지점명'], searched_reviews['유저'], searched_reviews['메뉴'], searched_reviews['리뷰'],
                                    searched_reviews['총점'], searched_reviews['맛'], searched_reviews['양'], searched_reviews['배달'], searched_reviews['시간']], axis=1)

        return(sent_topics_df)

    df_topic_sents_keywords = format_topics_sentences(
        ldamodel=optimal_model)
    df_topic_review = df_topic_sents_keywords.reset_index()
    df_topic_review.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib',
                               'Topic_Label', 'Keywords', '지점명', '유저', '메뉴', '리뷰', '총점', '맛', '양', '배달', '시간']
    df_topic_review

    # 토픽별 리뷰수
    element_count = {}
    for row in df_topic_review.iloc:
        element_count.setdefault(row['Dominant_Topic'], 0)
        element_count[row['Dominant_Topic']] += 1

    # 토픽별 총점 합계
    topic_total = {}
    for row in df_topic_review.iloc:
        topic_total.setdefault(row['Dominant_Topic'], 0)
        topic_total[row['Dominant_Topic']] += row['총점']

    # 토픽별 평균 총점
    for topic in topic_total:
        topic_total[topic] = topic_total[topic] / element_count[topic]
    topic_total

    # 평균 별점 순으로 정렬
    sdict = sorted(topic_total.items(),
                   key=lambda x: x[1], reverse=True)

    element_count = {}

    for item in searched_reviews['지점명']:
        element_count.setdefault(item, 0)
        element_count[item] += 1

    result = []
    for i in range(3):
        topic = int(sdict[i][0])

        condition = df_topic_review['Dominant_Topic'] == topic
        topdf = df_topic_review[condition]
        element_count = {}

        for item in topdf['지점명']:
            element_count.setdefault(item, 0)
            element_count[item] += 1

        rs = list(element_count.items())
        rs.sort(key=lambda x: x[1], reverse=True)
        best_rest = rs[0]
        score = round(sdict[i][1], 4)
        res = optimal_model.get_topic_words(topic, top_n=10)
        label = labels_per_topic[topic]
        keywords = ', '.join(w for w, p in res)
        result.append([label, keywords, score, best_rest])

    # topics => [토픽번호, 레이블, 키워드와 분포리스트]
    # result => [레이블, 키워드리스트, 스코어, 음식점]
    return [topics, result]


def topic_modeling2(searched_reviews):
    # 데이터 프레임의 '리뷰' 열의 값들을 str 형식으로 바꾸기
    searched_reviews.리뷰 = searched_reviews.리뷰.astype(str)

    # 중복데이터 삭제
    searched_reviews.drop_duplicates(subset=['리뷰'], inplace=True)

    # 한글이 아니면 빈 문자열로 바꾸기
    searched_reviews['리뷰'] = searched_reviews['리뷰'].str.replace(
        '[^ㄱ-ㅎㅏ-ㅣ가-힣]', ' ', regex=True)

    # 빈 문자열 NAN 으로 바꾸기
    searched_reviews = searched_reviews.replace({'': np.nan})
    searched_reviews = searched_reviews.replace(
        r'^\s*$', np.nan, regex=True)

    # NAN 이 있는 행 삭제
    searched_reviews.dropna(how='any', inplace=True)

    # 인덱스 차곡차곡
    searched_reviews = searched_reviews.reset_index(drop=True)

    # 리뷰 데이터를 리스트로 변환
    Data_list = searched_reviews.리뷰.values.tolist()
    # 정규화 처리
    Data_list = list(map(okt.normalize, Data_list))

    # 명사,형용사 추출
    data_word = []

    # 한글자 짜리는 모두 없애야 함
    stopword = ['같다', '이다', '있다', '여기', '항상', '완전',
                '정말', '너무', '보고', '오늘', '역시', '이번', '다음', '아주']

    for i, document in enumerate(Data_list):
        clean_words = []
        for word in okt.pos(document, stem=True):  # 어간 추출
            if word[1] in ['Noun', 'Adjective']:
                if len(word[0]) >= 2 and word[0] not in stopword:
                    clean_words.append(word[0])
        data_word.append(clean_words)

    def compute_coherence_values(data_word, limit, start=4, step=2):

        coherence_values = []
        model_list = []
        for num_topics in range(start, limit, step):
            model = tp.LDAModel(
                k=num_topics, alpha=0.1, eta=0.01, min_cf=1, tw=tp.TermWeight.PMI, rm_top=1)
            for i, line in enumerate(data_word):
                if not line:
                    line.append(" ")
                model.add_doc(line)
            model.train(500)  # 학습 정도
            model_list.append(model)
            coherence_model = ch.Coherence(
                model, coherence='c_v', window_size=0, targets=None, top_n=10, eps=1e-12, gamma=1.0)
            coherence_values.append(coherence_model.get_score())
        return model_list, coherence_values

    model_list, coherence_values = compute_coherence_values(
        data_word=data_word, start=4, limit=21, step=2)
    # 최적의 모델 얻기
    limit = 21
    start = 4
    step = 2
    x = range(start, limit, step)
    topic_num = 0
    count = 0
    max_coherence = 0

    for m, cv in zip(x, coherence_values):
        coherence = cv
        if coherence >= max_coherence:
            max_coherence = coherence
            topic_num = m
            model_list_num = count
        count = count+1

    # 최적의 모델
    optimal_model = model_list[model_list_num]
    # 주제 단어 후보군 뽑기
    extractor = tp.label.PMIExtractor(
        min_cf=1, min_df=1, min_len=2, max_len=5, max_cand=10000)
    cands = extractor.extract(optimal_model)

    labels_per_topic = []

    labeler = tp.label.FoRelevance(
        optimal_model, cands, min_df=1, smoothing=0.01, mu=0.25)

    topics = []
    for k in range(optimal_model.k):
        labels_per_topic.append(labeler.get_topic_labels(k, top_n=2)[
                                0][0]+' + '+labeler.get_topic_labels(k, top_n=2)[1][0])
        keywords = []
        for word, prob in optimal_model.get_topic_words(k, top_n=10):
            keywords.append([word, prob])
        topics.append([k, labeler.get_topic_labels(k, top_n=2)[
            0][0]+' + '+labeler.get_topic_labels(k, top_n=2)[1][0], keywords])

    def format_topics_sentences(ldamodel=optimal_model):
        new_doc = ldamodel.docs
        sent_topics_df = pd.DataFrame()

        for i in range(len(data_word)):
            topics = new_doc[i].get_topics()
            for j, (topic_num, prop_topic) in enumerate(topics):
                if j == 0:
                    wp = optimal_model.get_topic_words(
                        topic_num, top_n=10)
                    topic_keywords = ", ".join(
                        [word for word, prop in wp])
                    topic_label = labels_per_topic[topics[j][0]]
                    sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(
                        prop_topic, 4), topic_label, topic_keywords]), ignore_index=True)
                else:
                    break
        sent_topics_df.columns = [
            'Dominant_Topic', 'Perc_Contribution', 'Topic_Label', 'Topic_Keywords']
        sent_topics_df = pd.concat([sent_topics_df, searched_reviews['지점명'], searched_reviews['유저'], searched_reviews['메뉴'], searched_reviews['리뷰'],
                                    searched_reviews['총점'], searched_reviews['맛'], searched_reviews['양'], searched_reviews['배달'], searched_reviews['시간']], axis=1)

        return(sent_topics_df)

    df_topic_sents_keywords = format_topics_sentences(
        ldamodel=optimal_model)
    df_topic_review = df_topic_sents_keywords.reset_index()
    df_topic_review.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib',
                               'Topic_Label', 'Keywords', '지점명', '유저', '메뉴', '리뷰', '총점', '맛', '양', '배달', '시간']
    df_topic_review

    # 토픽별 리뷰수
    element_count = {}
    for row in df_topic_review.iloc:
        element_count.setdefault(row['Dominant_Topic'], 0)
        element_count[row['Dominant_Topic']] += 1

    # 토픽별 총점 합계
    topic_total = {}
    for row in df_topic_review.iloc:
        topic_total.setdefault(row['Dominant_Topic'], 0)
        topic_total[row['Dominant_Topic']] += row['총점']

    # 토픽별 평균 총점
    for topic in topic_total:
        topic_total[topic] = topic_total[topic] / element_count[topic]
    topic_total

    # 평균 별점 순으로 정렬
    sdict = sorted(topic_total.items(),
                   key=lambda x: x[1], reverse=True)

    element_count = {}

    for item in searched_reviews['지점명']:
        element_count.setdefault(item, 0)
        element_count[item] += 1

    result = []
    for i in range(3):
        topic = int(sdict[i][0])

        condition = df_topic_review['Dominant_Topic'] == topic
        topdf = df_topic_review[condition]
        element_count = {}

        for item in topdf['지점명']:
            element_count.setdefault(item, 0)
            element_count[item] += 1

        rs = list(element_count.items())
        rs.sort(key=lambda x: x[1], reverse=True)
        best_rest = rs[0]
        score = round(sdict[i][1], 4)
        res = optimal_model.get_topic_words(topic, top_n=10)
        label = labels_per_topic[topic]
        keywords = ', '.join(w for w, p in res)
        result.append([label, keywords, score, best_rest])

    # topics => [토픽번호, 레이블, 키워드와 분포리스트]
    # result => [레이블, 키워드리스트, 스코어, 음식점]
    return [topics, result]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/home', methods=('GET', 'POST'))
def home():
    if request.method == 'GET':
        location = request.args.get('location')
        session['location'] = location

        data = pd.read_excel(review_file[location])
        element_count = {}

        for item in data['지점명']:
            element_count.setdefault(item, 0)
            element_count[item] += 1

        return render_template('home.html', location=location, restaurant_list=list(element_count.keys()))


@app.route('/findByMenu', methods=('GET', 'POST'))
def menu_result():
    if request.method == 'GET':
        location = session['location']

        data = pd.read_excel(review_file[location])
        user_input = request.args.get("keyword")
        arr = []
        for i in range(len(data['메뉴'])):
            if type(data['메뉴'][i]) == str and user_input in data['메뉴'][i]:
                arr.append(data.loc[i])

        searched_reviews = pd.DataFrame(
            arr, columns=['지점명', '유저', '메뉴', '리뷰', '총점', '맛', '양', '배달', '시간'])

        topics, result = topic_modeling(searched_reviews)
        return render_template('result_menu.html', location=location, menu=user_input, topics=topics, result=result)


@app.route('/findByRestaurant/<restaurant>', methods=('GET', 'POST'))
def rest_result(restaurant):
    if request.method == 'GET':
        location = session['location']

        data = pd.read_excel(review_file[location])
        user_input = restaurant
        arr = []

        for i in range(len(data['지점명'])):
            if type(data['지점명'][i]) == str and user_input in data['지점명'][i]:
                arr.append(data.loc[i])

        searched_reviews = pd.DataFrame(
            arr, columns=['지점명', '유저', '메뉴', '리뷰', '총점', '맛', '양', '배달', '시간'])
        searched_reviews_bad = searched_reviews[(searched_reviews.총점 <= 2)]
        topics, result = topic_modeling(searched_reviews)
        topics2, result2 = topic_modeling2(searched_reviews_bad)
        return render_template('result_rest.html', location=location, restaurant=restaurant, topics=topics, result=result, topics2=topics2, result2=result2)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port="5000", debug=True)
