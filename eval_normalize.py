#encoding:utf-8

import os
import pymysql as db
import jieba
import tensorflow as tf
import oss2

def main():
    try:
        conn = db.Connect(host='139.196.132.213', 
                          user='luzhaohui', passwd='zhongba@01', db='zhongba', port=3306, charset='utf8')
        
        #检索需要评估的文章
        cur = conn.cursor()
        rowcount = cur.execute('select id, title, content from ai_texts ORDER BY created_at DESC LIMIT 0,100')
        print("共有{:d}篇文章待评估。\n".format(rowcount))

        results=cur.fetchall()
        for row in results:
            id = str(row[0])
            title = row[1]
            content = split_words(row[2])
            create_tfrecord_file(id, title, content)
            
        cur.close()
                
        conn.close()
    except db.Error as e:
        print("Mysql Error {:d}:{:s}\n".format(e.args[0], e.args[1]))

def split_words(text):
    seg_list = jieba.cut(text, cut_all=False)
    return ' '.join(seg_list)

def create_tfrecord_file(id, title, content):
#    local_path_prefix = 'd:/tmp/tfrecord_files/eval.tfrecords/'
    local_path_prefix = 'D:/ai/cnn-text-classification-tf/data/eval.tfrecords'
    oss_path_prefix = 'text-cnn/eval.tfrecords/'


    local_filepath = os.path.join(local_path_prefix, id)
    oss_filepath = os.path.join(oss_path_prefix, id)

    id_raw = id.encode(encoding='utf-8')
    title_raw = title.encode(encoding='utf-8')
    content_raw = content.encode(encoding='utf-8')
    
    example = tf.train.Example(
            features=tf.train.Features(feature={
                    'id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[id_raw])),
                    'title': tf.train.Feature(bytes_list=tf.train.BytesList(value=[title_raw])), 
                    'text_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[content_raw]))
                    }))

    #先写到本地临时目录
    writer = tf.python_io.TFRecordWriter(local_filepath)
    writer.write(example.SerializeToString())
    writer.close()
    
#    #上传至oss
#    bucket = oss2.Bucket(oss2.Auth(access_key_id, access_key_secret), endpoint, bucket_name)
#    bucket.put_object_from_file(oss_filepath, local_filepath)
#    
#    #删除本地临时文件
#    os.remove(local_filepath)

access_key_id = 'LTAICj9Xwo2Tm5L4'
access_key_secret = 'LTUo4ko7qnN7xV7AkNr6vaDeEcbL9P'
bucket_name = 'myaitest001'
endpoint = 'http://oss-cn-shanghai.aliyuncs.com'
if __name__ == '__main__':
    main()
