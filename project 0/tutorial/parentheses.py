class Stack:
    def __init__(self):
        self.List = []
        self.Count = 0
        
    def Empty(self):
        return self.Count == 0 
    
    def Pop(self):
        if self.Empty() :
            print("No item")
        else:
            self.Count-=1 
            return self.List.pop()  
    
    def Push(self, item):
        self.List.append(item)
        self.Count+=1
        

parentheses_strings = ["()[{}]()","(){}[]","{{{{{{{[]}}}}}}","[][]{{{(((0))}}}","({}((}))]{]})"]

def balance_check(parentheses_strings):
    ParStack = Stack()
    for parentheses in parentheses_strings:
        if parentheses in '({[':  
            
            ParStack.Push(parentheses)
            continue
            
        elif parentheses in ')}]':
            
            if ParStack.Empty(): 
                return 0
            Poped_Parentheses = ParStack.Pop()
            if (parentheses == ')' and Poped_Parentheses == '(') or (parentheses == '}' and Poped_Parentheses == '{')  or (parentheses == ']' and Poped_Parentheses == '[') :
                continue
            return 0
        
        return 2
    
    if ParStack.Empty(): 
        return 1
    return 0
        
    
def main():
    for String in parentheses_strings:
        x = balance_check(String)
        if x == 1: 
            print("The string is balanced\n")
        elif x == 0: 
            print("The string is not balanced\n")
        else: 
            print("WRONG INPUT: Only parentheses are allowed\n")
    

if __name__ == "__main__":
    main()