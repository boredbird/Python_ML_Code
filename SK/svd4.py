def svd(mat, feature, steps=500, gama=0.02, lamda=0.3):
    slowRate = 0.99
    preRmse = 1000000000.0
    nowRmse = 0.0

    user_feature = numpy.matrix(numpy.random.rand(mat.shape[0], feature))
    item_feature = numpy.matrix(numpy.random.rand(mat.shape[1], feature))

    for step in range(steps):
        rmse = 0.0  
        n = 0  
        for u in range(mat.shape[0]):
            for i in range(mat.shape[1]):
                if not numpy.isnan(mat[u,i]):
                    pui = float(numpy.dot(user_feature[u,:], item_feature[i,:].T))
                    eui = mat[u,i] - pui
                    rmse += pow(eui, 2)
                    n += 1 
                    for k in range(feature):
                        user_feature[u,k] += gama*(eui*item_feature[i,k] - lamda*user_feature[u,k])
                        item_feature[i,k] += gama*(eui*user_feature[u,k] - lamda*item_feature[i,k]) # 原blog这里有错误 

        nowRmse = sqrt(rmse * 1.0 / n)
        print 'step: %d      Rmse: %s' % ((step+1), nowRmse)
        if (nowRmse < preRmse):  
            preRmse = nowRmse
        else:
            break # 这个退出条件其实还有点问题
        gama *= slowRate
        step += 1

    return user_feature, item_feature
