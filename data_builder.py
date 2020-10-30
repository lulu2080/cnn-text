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
        
        #文章等级：优
        cur = conn.cursor()
        rowcount = cur.execute('select id, title, content from ai_texts where view_count >= 1000000')
        print("共有{:d}篇阅读数>=100万的文章。\n".format(rowcount))
        train_count = int(rowcount * 0.8)

        counter = 0
        results=cur.fetchall()
        for row in results:
            id = row[0]
            title = row[1]
            content = split_words(row[2])
            
#            print("{}\n".format(title))
            counter += 1;
            if counter <= train_count:
                create_tfrecord_file(id, title, content, 3, 0)
            else:
                create_tfrecord_file(id, title, content, 3, 1)
        cur.close()
        
        #文章等级：良
        cur = conn.cursor()
        rowcount = cur.execute('select id, title, content from ai_texts where view_count >= 500000 and view_count < 1000000')
        print("共有{:d}篇阅读数>=50万且<100万的文章。\n".format(rowcount))
        train_count = int(rowcount * 0.8)

        counter = 0
        results=cur.fetchall()
        for row in results:
            id = row[0]
            title = row[1]
            content = split_words(row[2])
            
#            print("{}\n".format(title))
            counter += 1;
            if counter <= train_count:
                create_tfrecord_file(id, title, content, 2, 0)
            else:
                create_tfrecord_file(id, title, content, 2, 1)
        cur.close()

        #文章等级：中
        cur = conn.cursor()
        rowcount = cur.execute('select id, title, content from ai_texts where view_count >= 100000 and view_count < 500000 limit 0,1500')
        print("共有{:d}篇阅读数>=10万且<50万的文章。\n".format(rowcount))
        train_count = int(rowcount * 0.8)

        counter = 0
        results=cur.fetchall()
        for row in results:
            id = row[0]
            title = row[1]
            content = split_words(row[2])
            
#            print("{}\n".format(title))
            counter += 1;
            if counter <= train_count:
                create_tfrecord_file(id, title, content, 1, 0)
            else:
                create_tfrecord_file(id, title, content, 1, 1)
        cur.close()

        #文章等级：差
        cur = conn.cursor()
        rowcount = cur.execute('select id, title, content from ai_texts where view_count <= 50000 limit 0,2000')
        print("共有{:d}篇阅读数<=5万的文章。\n".format(rowcount))
        train_count = int(rowcount * 0.8)

        counter = 0
        results=cur.fetchall()
        for row in results:
            id = row[0]
            title = row[1]
            content = split_words(row[2])
            
#            print("{}\n".format(title))
            counter += 1;
            if counter <= train_count:
                create_tfrecord_file(id, title, content, 0, 0)
            else:
                create_tfrecord_file(id, title, content, 0, 1)
        cur.close()
        
        conn.close()
    except db.Error as e:
        print("Mysql Error {:d}:{:s}\n".format(e.args[0], e.args[1]))

def split_words(text):
    seg_list = jieba.cut(text, cut_all=False)
    return ' '.join(seg_list)

def create_tfrecord_file(id, title, content, label, purpose):
    if purpose == 0:
        local_path_prefix = 'd:/tmp/tfrecord_files/train.tfrecords/'
        oss_path_prefix = 'text-cnn/train.tfrecords/'
    else:
        local_path_prefix = 'd:/tmp/tfrecord_files/test.tfrecords/'
        oss_path_prefix = 'text-cnn/test.tfrecords/'

    local_filepath = os.path.join(local_path_prefix, str(id))
    oss_filepath = os.path.join(oss_path_prefix, str(id))

    title_raw = title.encode(encoding='utf-8')
    content_raw = content.encode(encoding='utf-8')
    example = tf.train.Example(
            features=tf.train.Features(feature={
                    'title': tf.train.Feature(bytes_list=tf.train.BytesList(value=[title_raw])), 
                    'text_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[content_raw])), 
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                    }))

    #先写到本地临时目录
    writer = tf.python_io.TFRecordWriter(local_filepath)
    writer.write(example.SerializeToString())
    writer.close()
    
    #上传至oss
    bucket = oss2.Bucket(oss2.Auth(access_key_id, access_key_secret), endpoint, bucket_name)
    bucket.put_object_from_file(oss_filepath, local_filepath)
    
    #删除本地临时文件
    os.remove(local_filepath)

access_key_id = ''
access_key_secret = ''
bucket_name = ''
endpoint = ''
if __name__ == '__main__':
    main()
