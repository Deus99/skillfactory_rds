import requests
from bs4 import BeautifulSoup
from threading import Thread
import queue
import time
import pandas as pd



def worker(url, queue):  # get queue as argument
    car_links_all = []
    response = requests.get(url)
    #print(response)
    page = BeautifulSoup(response.text, 'html.parser')
    car_links = page.find_all('a', class_='Link ListingItemTitle-module__link')
    for link in car_links:
        car_links_all.append(link.get('href'))
        # send result to main thread using queue
    queue.put(car_links_all)

def extract_features(url, queue):  # get queue as argument
    r = requests.get(url)
    print(r)
    encoding = r.encoding if 'charset' in r.headers.get('content-type', '').lower() else None
    #soup = BeautifulSoup(r.content, from_encoding=encoding)
    soup = BeautifulSoup(r.content, 'html.parser')
    dict_car = {}
    list1 = ['bodyType', 'brand', 'color', 'fuelType', 'modelDate', 'name', \
             'numberOfDoors', 'productionDate', 'vehicleConfiguration', \
             'vehicleTransmission', 'engineDisplacement', 'enginePower', \
             'description']
    list2 = ['mileage', 'Привод', 'Руль', 'Состояние', \
             'Владельцы', 'ПТС', 'Владение', 'Таможня']
    list2_numb = ['CardInfo__row CardInfo__row_kmAge', 'CardInfo__row CardInfo__row_drive',
                  'CardInfo__row CardInfo__row_wheel', 'CardInfo__row CardInfo__row_state',
                  'CardInfo__row CardInfo__row_ownersCount', 'CardInfo__row CardInfo__row_pts',
                  'CardInfo__row CardInfo__row_owningTime', 'CardInfo__row CardInfo__row_customs']
    for i in list1:
        try:
            dict_car[i] = soup.find('meta', itemprop=i).get('content')
        except:
            dict_car[i] = ''
    try:
        #options = [i.text for i in soup.find_all('li', class_='CardComplectation__itemContentEl')]
        #opt_str = '|'.join(options)
        #dict_car['Комплектация'] = opt_str
        dict_car['Комплектация'] = '|'.join([i.text for i in soup.find_all(
            'li', class_='CardComplectation__itemContentEl')])
    except:
        dict_car['Комплектация'] = ''

    for i, k in enumerate(list2):
        try:
            dict_car[k] = soup.find('li', class_=list2_numb[i]).find_all('span', class_='CardInfo__cell')[1].text
        except:
            dict_car[k] = ''
    dict_car['id'] = url
    print(dict_car)
    new_df = pd.DataFrame(dict_car, columns=dict_car.keys(), index=[0])
    queue.put(new_df)

def pageparser_mt(links):
    # от 0 страницы до 200 с шагом 50 - для multithreading
    all_results = pd.DataFrame()
    len_car = len(links)

    for j in range(0, len_car, 12):
 #       print(j)
        time.sleep(0.25)
        # all_links = [
        #    'https://auto.ru/cars/used/?page=' + str(i) for i in range(j, j + 50)]
        all_links = links[j:j+12]

        all_threads = []
        my_queue = queue.Queue()

        # run threads
        for url in all_links:
#            print(url)
            t = Thread(target=extract_features, args=(url, my_queue))
            t.start()
            all_threads.append(t)

        for i in range(1, len(all_links)):
            data = my_queue.get()
#            print(f'итерация {i}, данные {data}')

            all_results = pd.concat([all_results, data]).drop_duplicates()
#            print(all_results.info())
        print ('stage complete')


            # if len(res_upd) > len(all_results):
            #    all_results = res_upd
            # else:
            #    break

    return all_results

def linkparser_mt(year, brand):
    # от 0 страницы до 200 с шагом 50 - для multithreading
    all_results = []

    for j in range(1, 100, 50):
        time.sleep(0.5)
        all_links = [
            'https://auto.ru/cars/' + brand + '/used/?year_from=' + str(year) + '&year_to=' +
            str(year-2) + '&page=' + str(i) for i in range(j, j + 50)]


        all_threads = []
        my_queue = queue.Queue()

        # run threads
        for url in all_links:
#            print(url)
            t = Thread(target=worker, args=(url, my_queue))
            t.start()
            all_threads.append(t)

        for i in range(1, 50 + 1):
            data = my_queue.get()

            res_upd = list(set(all_results))
            new = len(res_upd) - len(all_results)
            print(f'added {new} cars {brand} of {year}')

            all_results.extend(data)
            pd.Series(all_results).to_csv('links\\' + str(brand) + '_' + str(year) + '.csv')
#            print(len(all_results))

    print(f'parsing {brand} of {year} complete')
    return list(set(all_results))