import tensorflow as tf
import kdd99_input_data
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer
#绘图用
x_plot=[];y_plot=[]
a_x_plot=[];a_y_plot=[]
b_x_plot=[];b_y_plot=[]
attack_count_x_plot = [];attack_count_y_plot = []

def easy_ann_model(times):

    # 构建模型
    x = tf.placeholder("float", [None, 41])
    W = tf.Variable(tf.zeros([41, 23]))
    b = tf.Variable(tf.zeros([23]))

    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # 评价指标
    y_ = tf.placeholder("float", [None, 23])
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

    # 开始训练
    # 便于释放资源
    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)

        batch_xs, batch_ys = kdd99_input_data.get_data_ann(filename='train_data',num = 200)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        # 验证正确率
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        x_check, y_check = kdd99_input_data.get_data(filename="check_data",num=1000,offset = 0)
        accuracy = sess.run(accuracy, feed_dict={x: x_check, y_: y_check})
        return accuracy

def paint(x,y,text):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # correctly show Chinese
    plt.rcParams['axes.unicode_minus'] = False  # correctly show minus and positive
    plt.figure(figsize=(7, 5))

    plt.plot(x,y,'b-')
    plt.title(text)
    plt.grid(True)
    plt.show()

def svm_model(test_num = 10000):
    check_num = 500
    x_train , y_train = kdd99_input_data.get_data_svm_collect(num = test_num , random_option = True)
    pca = PCA(n_components=9,copy=True)
    pca.fit(x_train)

    pca_x_train = pca.transform(x_train).tolist()

    clf = svm.SVC(C=10 , kernel = 'rbf' , degree = 3)
    clf.fit(pca_x_train,y_train)

    x_check , y_check = kdd99_input_data.get_data_svm(filename = 'check_data' , num = check_num,random_option = True)
    pca_x_check = pca.transform(x_check).tolist()

    correct_count = 0
    a_wrong = 0# 弃真错误
    b_wrong = 0# 取伪错误
    attack_count = 0
    attack_type = []
    for index,item in enumerate(pca_x_check):
        result = clf.predict(item)
        if result[0] != '0':
            attack_count += 1
            if result[0] not in attack_type:
                attack_type.append(result[0])

        if result[0] == y_check[index][0]:
            correct_count += 1
        else :
            # 统计弃真和取伪错误
            if result[0] == '0':
                b_wrong += 1
            else:
                a_wrong += 1
    print(attack_type)
    return correct_count/check_num,a_wrong,b_wrong,attack_count

def show_accuracy(func , start , stop , step):
    for i in range(start , stop):
        result = func(i*step)
        x_plot.append(i)
        y_plot.append(result[0])

        a_x_plot.append(i)
        a_y_plot.append(result[1])

        b_x_plot.append(i)
        b_y_plot.append(result[2])

        attack_count_x_plot.append(i)
        attack_count_y_plot.append(result[3])

    paint(x_plot,y_plot,'accuracy')
    paint(a_x_plot,a_y_plot,'a wrong')
    paint(b_x_plot,b_y_plot,'b wrong')
    paint(attack_count_x_plot,attack_count_y_plot,'not correct count')

show_accuracy(svm_model,1,10,100)