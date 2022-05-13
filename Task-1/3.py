# Create a list to store the 6 letter words
sixLetterWords= []
# Open the file
with open('about.txt') as fin:
    # Read each line
    for line in fin.readlines():
        # Split the line into words
        for word in line.split(" "):
            # Check each word's length
            if len(word) == 6:
                # Add the 6 letter word to the list
                sixLetterWords.append(word)
# Print out the result
print(sixLetterWords)


count = 0;  
word = "";  
maxCount = 0;  
words = [];  
   
#Opens a file in read mode  
file = open("about.txt", "r")  
      
#Gets each line till end of file is reached  
for line in file:  
    #Splits each line into words  
    string = line.lower().replace(',','').replace('.','').split(" ");  
    #Adding all words generated in previous step into words  
    for s in string:  
        words.append(s);  
   
#Determine the most repeated word in a file  
for i in range(0, len(words)):  
    count = 1;  
    #Count each word in the file and store it in variable count  
    for j in range(i+1, len(words)):  
        if(words[i] == words[j]):  
            count = count + 1;  
              
    #If maxCount is less than count then store value of count in maxCount  
    #and corresponding word to variable word  
    if(count > maxCount):  
        maxCount = count;  
        word = words[i];  
          
print("Most repeated word: " + word);
file.close();  


