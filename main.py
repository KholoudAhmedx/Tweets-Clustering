import re
import string
import numpy as np
import matplotlib.pyplot as plt
import random as rd

def preprocessing_tweets(filePath):
    f = open(filePath, "r")
    tweets = list(f)
    tweetsList = []

    for i in range(len(tweets)):
        # Remove id and timestamp
        tweets[i] = tweets[i][50:]
        
        # Remove words with the @ symbol
        tweets[i] = re.sub(r"@\S+", "", tweets[i])

        # Remove Urls
        tweets[i] = re.sub(r"http\S+", "", tweets[i])

        # Remove hashtag symbols
        tweets[i] = re.sub(r"\#", "", tweets[i])

        # Convert every word to lowercase
        tweets[i] = tweets[i].lower()

        # Remove punctuation such as '', ?, !, :
        tweets[i] = tweets[i].translate(str.maketrans('', '', string.punctuation))

        # Remove extra/duplicate whitespaces and newline charachers
        tweets[i] =  " ".join(tweets[i].split())


        tweetsList.append(tweets[i].split())

    return tweetsList

def KMeans(tweets, k, max_iterations=50):
    
    centroids = initialize_random_centorid(k)

    iter_count = 0 
    old_centroids = []

    # Kmeans runs until not converged and or until maximum iterations not reached
    while(is_converged(old_centroids, centroids) == False and (iter_count < max_iterations)):

        print("iteration number: " + str(iter_count))

        clusters = createCluster(tweets, centroids)

        old_centroids = centroids

        centroids = update_centroids(clusters)
        iter_count = iter_count  + 1

    if iter_count == max_iterations:
        print("maximum iterations reached, kmeans not converged")
    else:
        print("converged")

    # compute sse
    sse = compute_sse(clusters)
    return clusters, sse    


def is_converged(old_centroids, new_centroids):

    if(len(old_centroids) != len(new_centroids)):
        return False

    for c in range(len(new_centroids)):
    
        dis = jaccard_distance(old_centroids[c], new_centroids[c])
        if dis  == 0:
            return True
        else:
            return False           

def initialize_random_centorid(k):
    #initialize random centorid
    centorids = []
    
    for i in range(k):

        random_tweet_indxs = np.random.randint(0, len(tweets) - 1)
        centorids.append(tweets[random_tweet_indxs])

    return centorids

def createCluster(tweets, centroids):
    
    clusters = {}

    # for centroid in range(len(centroids)):
    #     clusters.append([])

    for tweet in range(len(tweets)):  
          
        min_dis = 10000
        cluster_index = -1

        for centroid in range(len(centroids)):
            dis = jaccard_distance(centroids[centroid], tweets[tweet])
            if centroids[centroid] == tweets[tweet]:
                min_dis = 0
                cluster_index = centroid
                break

            if dis < min_dis:
                min_dis = dis
                cluster_index = centroid   
            
        if min_dis == 1.0:
            cluster_index =rd.randint(0, len(centroids) - 1)
        
        # create empty list for each tweet to append the tweets
        clusters.setdefault(cluster_index, []).append([tweets[tweet]])
        
        # Keep track of the last tweet index 
        last_tweet_index = len(clusters.setdefault(cluster_index, [])) - 1
        
        # append minimum distance between each tweet and the centroids to compute sse
        clusters.setdefault(cluster_index,[])[last_tweet_index].append(min_dis)

    return clusters


def update_centroids(clusters):
   
    centorids = []
   
    for c in range(len(clusters)):
        minimum_distance_sum = 10000
        centroid_index = -1
        
        # store minimimum distances between each tweet and other tweets 
        min_dis_list = []

        for t1 in range(len(clusters[c])):
            #create empty list for each tweet to store minimum distances
            min_dis_list.append([])
            dis_sum =  0

            for t2 in range(len(clusters[c])):
                
                if t1 != t2:
                    dis = jaccard_distance(clusters[c][t1][0], clusters[c][t2][0])
                    min_dis_list[t1].append(dis)
                    dis_sum += dis
                else:
                    min_dis_list[t1].append(0)

                #select the tweet with the minimum distance sum to be the new centorid
            if dis_sum < minimum_distance_sum:
                minimum_distance_sum = dis_sum
                centroid_index = t1

        centorids.append(clusters[c][centroid_index][0])
    return centorids    

# It is small if tweet A and B are similar.
# It is large if they are not similar.
# It is 0 if they are the same.  
# It is 1 if they are completely different.
def jaccard_distance(tweets1, tweets2):
    return 1 - (len(set(tweets1).intersection(tweets2))/len(set().union(tweets1, tweets2)))

def compute_sse(clusters):

    sse = 0

    # iterate every cluster 'c', compute SSE as the sum of square of distances of the tweet from it's centroid
    for cluster in range(len(clusters)):
        for tweet in range(len(clusters[cluster])):
            sse = sse + (clusters[cluster][tweet][1] ** 2)
 
    return sse

if __name__ == '__main__':

    data_url = 'Health-Tweets/bbchealth.txt'
    tweets = preprocessing_tweets(data_url)
    
    # Default values of k and no. of experiments
    experiment = 5
    k = 1
    
    # x = input("Enter 1 for entering k value, 2 for default: ")
    # if x == 1:
    #    k = input("Enter k value : ")
    # elif x == 2:
    #     k = 3
    

    k_values = []
    sse_values = []
    for e in range (experiment):
        print("running kmeans for experiment number " + str(e + 1) + "  for k = " +str(k))
        clusters , sse = KMeans(tweets, k)
        for c in range(len(clusters)):
            print(str(c+1) + "-> ", str(len(clusters[c])) + "  tweets")

        print("sse -> " + str(sse))
        sse_values.append(sse)
        print('\n')
        k_values.append(k)
        k = k + 1

    # Plotting
    plt.plot(k_values, sse_values)
    plt.xlabel('k values')
    plt.ylabel('sse values')    
    plt.show()


#     l = initialize_random_centorid(2)
#     print(l)
#     print("-----------------")
#     cen = createCluster(tweets, l)

#     u = update_centroids(cen)
#     print(u)
#     # u = update_centroids(cen)
#    # print(u)
    