def makeActions():
    acts = [lambda x,i=i: i ** x for i in range(5)]
    return acts

acts = makeActions()

print('0', acts[0](1))
print('1', acts[1](1))
print('2', acts[2](1))
print('3', acts[3](1))
print('4', acts[4](1))
