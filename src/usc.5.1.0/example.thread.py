import threading

def f(abc):
  abc[0]="a"

abc=[]
abc.append("abcd")

print(abc)
t=threading.Thread(target=f,args=(abc,))
t.start()
t.join()
print(abc)


