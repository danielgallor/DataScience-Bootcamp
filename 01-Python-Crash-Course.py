# Crash Course Exercises
#Q1
7**4
#Q2
S ="Hello my name is Sam"
S.split()
#Q3
planet = "Earth"
diameter = 12742
f"The diameter of {planet} is {diameter} kilometers."
#Q4
d = {'k1':[1,2,3,{'tricky':['oh','man','inception',{'target':[1,2,3,'hello']}]}]}
d['k1'][3]['tricky'][3]['target'][3]
#Q5
def domainget(email):
    return email.split('@')[-1]

domainget('user@domain.com')
#Q6
def findDog(string):
    return 'dog' in string.lower().split()
  
findDog('Is there a Dog here?')
#Q7
def countDog(string):
    count = 0
    a = string.lower().split()
    for i in a:
        if i =='dog':
            count += 1
    return count

countDog('This dog runs faster than the other dog dude!')
#Q8
seq = ['soup','dog','salad','cat','great']
list(filter(lambda word: word[0]=='s' ,seq))
#Final Problem
def caught_speeding(speed, is_birthday):
    if is_birthday:
        speeding = speed - 5    
    else:
        speeding = speed

       
    if speeding <= 60:
        print('No ticket')
    elif 61 <= speeding <= 80:
        print('Small Ticket')
    else:
        print('Big Ticket')
    
caught_speeding(81,True)
caught_speeding(81,False)
a=5
