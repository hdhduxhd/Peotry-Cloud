from neo_db.config import graph, CA_LIST, similar_words
from spider.show_profile import get_profile
import codecs
import os
import json
import base64

def query(name):
    print(name)
    data = graph.run(
    "match(p:诗人) -[r]->(n:`诗歌`) where p.name='%s' return p.name,n.name,r.relation limit 50" % (name)
    )
    data = list(data)
    return get_json_data(data)
def get_json_data(data):
    json_data={'data':[],"links":[]}
    d=[]
    
    
    for i in data:
        # print(i["p.Name"], i["r.relation"], i["n.Name"], i["p.cate"], i["n.cate"])
        d.append(i['p.name'])
        d.append(i['n.name'])
        d=list(set(d))
    name_dict={}
    count=0
    for j in d:
        data_item={}
        name_dict[j]=count
        count+=1
        data_item['name']=j
        json_data['data'].append(data_item)
    for i in data:
   
        link_item = {}
        
        link_item['source'] = name_dict[i['p.name']]
        
        link_item['target'] = name_dict[i['n.name']]
        link_item['value'] = i['r.relation']
        json_data['links'].append(link_item)

    return json_data
# f = codecs.open('./static/test_data.json','w','utf-8')
# f.write(json.dumps(json_data,  ensure_ascii=False))
def get_KGQA_answer(array):
    data_array=[]
    for i in range(len(array)-2):
        if i==0:
            name=array[0]
        else:
            name=data_array[-1]['p.name']
           
        data = graph.run(
            "match(p)-[r:%s{relation: '%s'}]->(n:Person{Name:'%s'}) return  p.name,n.name,r.relation" % (
                similar_words[array[i+1]], similar_words[array[i+1]], name)
        )
       
        data = list(data)
        print(data)
        data_array.extend(data)
        
        print("==="*36)
    with open("./spider/images/"+"%s.jpg" % (str(data_array[-1]['p.name'])), "rb") as image:
            base64_data = base64.b64encode(image.read())
            b=str(base64_data)
          
    return [get_json_data(data_array), get_profile(str(data_array[-1]['p.Name'])), b.split("'")[1]]
def get_answer_profile(name):
    with open("./spider/images/"+"%s.jpg" % (str(name)), "rb") as image:
        base64_data = base64.b64encode(image.read())
        b = str(base64_data)
    return [get_profile(str(name)), b.split("'")[1]]
        



