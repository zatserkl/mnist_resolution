import matplotlib.pyplot as plt
import numpy as np

from DataStore import DataStore
from ModelRes import ModelRes

data = DataStore()
# data.plotTrain()
print(data.X_train.shape)

# Fixed set of resolutions available
resolutions = (28, 14, 7)

model28 = ModelRes(28,
                   data.train28x28, data.y_train,
                   data.test28x28, data.y_test)
model28.train()

model14 = ModelRes(14,
                   data.train14x14, data.y_train,
                   data.test14x14, data.y_test)
model14.train()

model7 = ModelRes(7,
                  data.train7x7, data.y_train,
                  data.test7x7, data.y_test)
model7.train()

# Evaluate digits using softmax probability

model7x7 = model7.model
model14x14 = model14.model
model28x28 = model28.model

found = np.zeros(len(data.y_test), dtype=np.uint8) - 1
found7x7 = np.zeros(len(data.y_test), dtype=np.uint8) - 1
found14x14 = np.zeros(len(data.y_test), dtype=np.uint8) - 1
found28x28 = np.zeros(len(data.y_test), dtype=np.uint8) - 1
found28x28_all = np.zeros(len(data.y_test), dtype=np.uint8) - 1

threshold = 0.95

for ipic in range(data.y_test.shape[0]):
    # for ipic in range(10):
    # start from the resolution 7x7
    prob_vector = model7x7.predict_proba(model7.x_test[ipic:ipic+1, :])
    prob_max_index = np.argmax(prob_vector)
    prob_max = prob_vector[0, prob_max_index]
    # print('prob_vector:', prob_vector, 'prob_max:', prob_max,
    #       'prob_max_index:', prob_max_index, 'data.y_test[ipic]:', data.y_test[ipic])
    if prob_max >= threshold:
        found[ipic] = prob_max_index
        found7x7[ipic] = prob_max_index
        continue
    else:
        # check finer resolution 14x14
        prob_vector = model14x14.predict_proba(model14.x_test[ipic:ipic+1, :])
        prob_max_index = np.argmax(prob_vector)
        prob_max = prob_vector[0, prob_max_index]
        if prob_max >= threshold:
            found[ipic] = prob_max_index
            found14x14[ipic] = prob_max_index
            continue
        else:
            # check finer resolution 28x28
            prob_vector = model28x28.predict_proba(
                model28.x_test[ipic:ipic+1, :])
            prob_max_index = np.argmax(prob_vector)
            prob_max = prob_vector[0, prob_max_index]
            found[ipic] = prob_max_index    # append anyway: no other choice
            found28x28_all[ipic] = prob_max_index
            if prob_max >= threshold:
                found28x28[ipic] = prob_max_index

# accuracy
print('whole chain correct:', len(np.nonzero(found == data.y_test)[0]))
print('just 7x7 correct:', len(np.nonzero(found7x7 == data.y_test)[0]))
print('found14x14 correct:', len(np.nonzero(found14x14 == data.y_test)[0]))
print('found28x28 correct:', len(np.nonzero(found28x28 == data.y_test)[0]))
print('found28x28_all correct:', len(np.nonzero(found28x28_all == data.y_test)[0]))

print('whole chain incorrect:', len(np.nonzero(found != data.y_test)[0]))

# plot everything
plt.show()
