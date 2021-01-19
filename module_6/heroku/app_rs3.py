#!pip install streamlit


import streamlit as st
import numpy as np
import pandas as pd
import lightfm as lf
import nmslib
import pickle
import scipy.sparse as sparse



# Используемые функции

def nearest_items_nms(item_id, index, n=12):
    """Функция для поиска ближайших соседей, возвращает построенный индекс"""
    nbm = index.knnQuery(item_embeddings[item_id], k=n)
    return nbm

def get_items(nbm):
    """Возвращает 5 похожих товаров из данной категории в виде датафрейма"""
    df_goods = goods[goods.itemid.isin(nbm)].sort_values(by='f1_score', ascending=False).head(5)
    df_goods = df_goods[['itemid', 'brand', 'main_cat', 'fn_cat', 'price', 'rating']]
    return df_goods

def load_embeddings():
    """Функция для загрузки эмбендингов(векторных представлений) из файла"""
    with open('asin_embeddings.pickle', 'rb') as f:
        item_embeddings = pickle.load(f)

    # Тут мы используем nmslib, чтобы создать наш быстрый knn
    nms_idx = nmslib.init(method='hnsw', space='cosinesimil')
    nms_idx.addDataPointBatch(item_embeddings)
    nms_idx.createIndex(print_progress=True)
    return item_embeddings,nms_idx

def read_files():
    """Загружаем ранее созданный файл с товарами, рейтингами и пр."""
    goods = pd.read_csv('data/goods.csv')
    goods['title100'] = goods['title'].apply(lambda x: x[:100])
    return goods


def goods_top10(item_sel, item_sel2, item_sel3):
    sel_goods = goods[(goods.main_cat==item_sel) & (goods.sm_cat==item_sel2) & (
        goods.fn_cat==item_sel3)].sort_values(by='f1_score', ascending=False)[['itemid', 
        'fn_cat', 'price', 'buys', 'rating']].head(10)
    return sel_goods

def goods_str(item_sel, item_sel2, item_sel3):
    sel_goods = goods[(goods.main_cat==item_sel) & (goods.sm_cat==item_sel2) & (
        goods.fn_cat==item_sel3)]
    sel_goods['item_str'] = sel_goods['itemid'].astype(str) + '-' + sel_goods[
        'fn_cat'] + '- ' + sel_goods['title100']
    selected = sel_goods['item_str'].tolist()
    return selected

# Загружаем данные
goods = read_files()


#создаем словарь asin-itemid
#asins = goods[['asin','itemid']].set_index('asin').to_dict()['itemid']
item_embeddings, nms_idx = load_embeddings()






st.title('Рекомендательняа система по товарам Amazon')

main_goods = goods['main_cat'].unique().tolist()
item_sel = st.selectbox("Выберите основную категорию товара:", main_goods, 0)

sec_goods = goods[goods.main_cat==item_sel]['sm_cat'].unique().tolist()
item_sel2 = st.selectbox("Выберите категорию товара:", sec_goods, 0)

fin_goods = goods[(goods.main_cat==item_sel) & (goods.sm_cat==item_sel2)]['fn_cat'].unique().tolist()
item_sel3 = st.selectbox("Выберите тип товара:", fin_goods, 0)

sel_goods = goods_str(item_sel, item_sel2, item_sel3)
item_sel4 = st.selectbox('Выберете товар:', sel_goods, 0)

item_id = int(item_sel4.split('-')[0])


st.write ("Выбранный товар:")
st.write (item_sel4)

# получаем список индексов
index_nbm = nearest_items_nms(item_id, nms_idx)[0]

# получаем топ-5 товаров
goods_top5 = get_items(index_nbm)

st.write('Подобные товары:')
st.table(goods_top5)

#selected = goods_top10(item_sel, item_sel2, item_sel3)
#st.table(selected)


# форма для ввода id товара
#item_id = int(st.text_input('введите item id'))

#st.write(item_id, type(item_id))

# получаем список индексов
#index_nbm = nearest_items_nms(item_id, nms_idx)[0]

# получаем топ-5 товаров
#goods_top5 = get_items(index_nbm)

#st.write('', goods_top5)

#st.write('Вы искали товар: ')

#st.write(goods[goods.itemid==item_id][['itemid', 'brand', 'main_cat', 'fn_cat', 'price', 'rating']])

#st.write('Подобные товары:')

#st.table(goods_top5)