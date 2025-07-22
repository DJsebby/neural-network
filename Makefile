CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall

# Sources
SOURCES = main.cpp NeuralNetwork.cpp matrix.cpp \
          activations/activations.cpp \
          layers/denseLayer.cpp \
          mnist_loader/mnist_loader.cpp

# Headers (just for dependency tracking)
HEADERS = NeuralNetwork.h matrix.h \
          activations/activations.h \
          layers/denseLayer.h \
          mnist_loader/mnist_loader.h

OBJECTS = $(SOURCES:.cpp=.o)
EXEC = neural_net

all: $(EXEC)

$(EXEC): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(EXEC) a.out


