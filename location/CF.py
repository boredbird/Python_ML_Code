# This Python file uses the following encoding: utf-8

from pprint import pprint
from math import sqrt


# 根据txt生成数据字典
def make_data():
    result = {}
    f = open('data/u.data', 'r')
    lines = f.readlines()
    for line in lines:
        (userId, itemId, score, time) = line.strip().split("\t")
        if not result.has_key(userId):
            result[userId] = {}
        result[userId][itemId] = float(score)
    return result


# critics = make_data()
critics = {
    'Lisa Rose': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.5,
                  'Just My Luck': 3.0, 'Superman Returns': 3.5, 'You, Me and Dupree': 2.5,
                  'The Night Listener': 3.0},

    'Gene Seymour': {'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5,
                     'Just My Luck': 1.5, 'Superman Returns': 5.0, 'The Night Listener': 3.0,
                     'You, Me and Dupree': 3.5},

    'Michael Phillips': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.0,
                         'Superman Returns': 3.5, 'The Night Listener': 4.0},

    'Claudia Puig': {'Snakes on a Plane': 3.5, 'Just My Luck': 3.0,
                     'The Night Listener': 4.5, 'Superman Returns': 4.0,
                     'You, Me and Dupree': 2.5},

    'Mick LaSalle': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
                     'Just My Luck': 2.0, 'Superman Returns': 3.0, 'The Night Listener': 3.0,
                     'You, Me and Dupree': 2.0},

    'Jack Matthews': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
                      'The Night Listener': 3.0, 'Superman Returns': 5.0, 'You, Me and Dupree': 3.5},

    'Toby': {'Snakes on a Plane': 4.5, 'You, Me and Dupree': 1.0, 'Superman Returns': 4.0}}


# 欧几里得距离
def sim_distance(prefs, person1, person2):
    si = {}
    for itemId in prefs[person1]:
        if itemId in prefs[person2]:
            si[itemId] = 1
    # no same item
    if len(si) == 0: return 0
    sum_of_squares = 0.0

    # 计算距离
    # for item in si:
    #    sum_of_squares =  pow(prefs[person1][item] - prefs[person2][item],2) + sum_of_squares
    # sum_of_squares=sum([pow(prefs[person1][item]-prefs[person2][item],2) for item in prefs[person1] if item in prefs[person2]])
    sum_of_squares = sum([pow(prefs[person1][item] - prefs[person2][item], 2) for item in si])
    return 1 / (1 + sqrt(sum_of_squares))


# 皮尔逊相关度
def sim_pearson(prefs, p1, p2):
    si = {}
    for item in prefs[p1]:
        if item in prefs[p2]: si[item] = 1

    if len(si) == 0: return 0

    n = len(si)

    # 计算开始
    sum1 = sum([prefs[p1][it] for it in si])
    sum2 = sum([prefs[p2][it] for it in si])

    sum1Sq = sum([pow(prefs[p1][it], 2) for it in si])
    sum2Sq = sum([pow(prefs[p2][it], 2) for it in si])

    pSum = sum([prefs[p1][it] * prefs[p2][it] for it in si])

    num = pSum - (sum1 * sum2 / n)
    den = sqrt((sum1Sq - pow(sum1, 2) / n) * (sum2Sq - pow(sum2, 2) / n))
    # 计算结束

    if den == 0: return 0

    r = num / den

    return r


# 推荐用户
def topMatches(prefs, person, n=5, similarity=sim_distance):
    scores = [(similarity(prefs, person, other), other) for other in prefs if other != person]
    scores.sort()
    scores.reverse()
    return scores[0:n]


# 基于用户推荐物品
def getRecommendations(prefs, person, similarity=sim_pearson):
    totals = {}
    simSums = {}

    for other in prefs:
        if other == person:
            continue
        sim = similarity(prefs, person, other)
        # 去除负相关的用户
        if sim <= 0: continue
        for item in prefs[other]:
            if item in prefs[person]: continue
            totals.setdefault(item, 0)
            totals[item] += sim * prefs[other][item]
            simSums.setdefault(item, 0)
            simSums[item] += sim
    rankings = [(totals[item] / simSums[item], item) for item in totals]
    # rankings=[(total/simSums[item],item) for item,total in totals.items()]
    rankings.sort()
    rankings.reverse()
    return rankings


# 基于物品的列表
def transformPrefs(prefs):
    itemList = {}
    for person in prefs:
        for item in prefs[person]:
            if not itemList.has_key(item):
                itemList[item] = {}
                # result.setdefault(item,{})
            itemList[item][person] = prefs[person][item]
    return itemList


# 构建基于物品相似度数据集
def calculateSimilarItems(prefs, n=10):
    result = {}
    itemPrefs = transformPrefs(prefs)
    c = 0
    for item in itemPrefs:
        c += 1
        if c % 10 == 0: print "%d / %d" % (c, len(itemPrefs))
        scores = topMatches(itemPrefs, item, n=n, similarity=sim_distance)
        result[item] = scores
    return result


# 构建基于人的相似度数据集
def calculateSimilarUsers(prefs, n=10):
    result = {}
    c = 0
    for user in prefs:
        c += 1
        if c % 10 == 0: print "%d / %d" % (c, len(prefs))
        scores = topMatches(prefs, user, n=n, similarity=sim_distance)
        result[user] = scores
    return result


# 基于物品的推荐
def getRecommendedItems(prefs, itemMatch, user):
    userRatings = prefs[user]
    scores = {}
    totalSim = {}
    # Loop over items rated by this user
    for (item, rating) in userRatings.items():
        # Loop over items similar to this one
        for (similarity, item2) in itemMatch[item]:

            # Ignore if this user has already rated this item
            if item2 in userRatings: continue
            # Weighted sum of rating times similarity
            scores.setdefault(item2, 0)
            scores[item2] += similarity * rating
            # Sum of all the similarities
            totalSim.setdefault(item2, 0)
            totalSim[item2] += similarity

            # Divide each total score by total weighting to get an average
    rankings = [(score / totalSim[item], item) for item, score in scores.items()]

    # Return the rankings from highest to lowest
    rankings.sort()
    rankings.reverse()
    return rankings


# 将id替换为电影名 构成数据集
def loadMovieLens(path='data'):
    # Get movie titles
    movies = {}
    for line in open(path + '/u.item'):
        (id, title) = line.split('|')[0:2]
        movies[id] = title

    # Load data
    prefs = {}
    for line in open(path + '/u.data'):
        (user, movieid, rating, ts) = line.split('\t')
        prefs.setdefault(user, {})
        prefs[user][movies[movieid]] = float(rating)
    return prefs

    # 测试
    # print sim_distance( critics,'Lisa Rose', 'Gene Seymour')
    # print sim_pearson( critics,'Lisa Rose', 'Gene Seymour')


print topMatches(critics, 'Lisa Rose', 10)

# res = getRecommendations( critics , 'Michael Phillips')
# print res

# print len(transformPrefs( critics ))
# 基于物品推荐
# res = calculateSimilarItems( critics )
# print getRecommendedItems( critics,res,'2')
# 基于泰坦尼克号的相关电影的推荐
# res = transformPrefs( critics )
# print getRecommendations( res , '313')
# 格式化数据 载入电影名 构建数据集
# print loadMovieLens()
# 构建人相关度列表 对比时间
res = calculateSimilarUsers(critics)
# pprint(res)