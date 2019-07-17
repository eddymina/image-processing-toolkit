import os 

path = 'images/'
# Function to rename multiple files 
def main(): 
    i = 0
      
    for filename in os.listdir(path): 
 
        dst ="img_" + str(i) + ".jpg"
        src =path+ filename 
        dst =path+ dst 


           
        # rename() function will 
        # rename all the files 
        os.rename(src, dst) 
        i += 1
  
# Driver Code 
if __name__ == '__main__': 
      
    # Calling main() function 
    main() 