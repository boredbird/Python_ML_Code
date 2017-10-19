# -*- coding:utf-8 -*-
__author__ = 'chunhui.zhang'

class Student(object):
    count = 0
    books = []
    def __init__(self, name, age):
        self.name = name
        self.age = age
    pass


Student.books.extend(["python", "javascript"])
print "Student book list: %s" % Student.books
# class can add class attribute after class defination
Student.hobbies = ["reading", "jogging", "swimming"]
print "Student hobby list: %s" % Student.hobbies
print dir(Student)

print

wilber = Student("Wilber", 28)
print "%s is %d years old" % (wilber.name, wilber.age)
# class instance can add new attribute
# "gender" is the instance attribute only belongs to wilber
wilber.gender = "male"
print "%s is %s" % (wilber.name, wilber.gender)
# class instance can access class attribute
print dir(wilber)
wilber.books.append("C#")
print wilber.books

print

will = Student("Will", 27)
print "%s is %d years old" % (will.name, will.age)
# will shares the same class attribute with wilber
# will don't have the "gender" attribute that belongs to wilber
print dir(will)
print will.books

wilber = Student("Wilber", 28)

print "Student.count is wilber.count: ", Student.count is wilber.count
wilber.count = 1
print "Student.count is wilber.count: ", Student.count is wilber.count
print Student.__dict__
print wilber.__dict__
del wilber.count
print "Student.count is wilber.count: ", Student.count is wilber.count

print

wilber.count += 3
print "Student.count is wilber.count: ", Student.count is wilber.count
print Student.__dict__
print wilber.__dict__

del wilber.count
print

print "Student.books is wilber.books: ", Student.books is wilber.books
wilber.books = ["C#", "Python"]
print "Student.books is wilber.books: ", Student.books is wilber.books
print Student.__dict__
print wilber.__dict__
del wilber.books
print "Student.books is wilber.books: ", Student.books is wilber.books

print

wilber.books.append("CSS")
print "Student.books is wilber.books: ", Student.books is wilber.books
print Student.__dict__
print wilber.__dict__


class Student(object):
    '''
    this is a Student class
    '''
    count = 0
    books = []

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def printInstanceInfo(self):
        print "%s is %d years old" % (self.name, self.age)

    pass


wilber = Student("Wilber", 28)
wilber.printInstanceInfo()


class Student(object):
    '''
    this is a Student class
    '''
    count = 0
    books = []

    def __init__(self, name, age):
        self.name = name
        self.age = age

    @classmethod
    def printClassInfo(cls):
        print cls.__name__
        print dir(cls)

    pass


Student.printClassInfo()
wilber = Student("Wilber", 28)
wilber.printClassInfo()


class Student(object):
    '''
    this is a Student class
    '''
    count = 0
    books = []

    def __init__(self, name, age):
        self.name = name
        self.age = age

    @staticmethod
    def printClassAttr():
        print Student.count
        print Student.books

    pass


Student.printClassAttr()
wilber = Student("Wilber", 28)
wilber.printClassAttr()


# 一个装饰器
def document_it(func):
    def new_function(*args, **kwargs):
        print("Runing function: ", func.__name__)
        print("Positional arguments: ", args)
        print("Keyword arguments: ", kwargs)
        result = func(*args, **kwargs)
        print("Result: ", result)
        return result

    return new_function


# 人工赋值
def add_ints(a, b):
    return a + b

#方式一 人工对装饰器赋值
cooler_add_ints = document_it(add_ints)
cooler_add_ints(3, 5)


#方式二 函数器前加装饰器名字
@document_it
def add_ints(a, b):
    return a + b




a = InfoValue()
