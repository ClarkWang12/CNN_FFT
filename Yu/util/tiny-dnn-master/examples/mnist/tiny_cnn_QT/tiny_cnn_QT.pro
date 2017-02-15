TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
    main.cpp

INCLUDEPATH += /home/clark/Documents/Codes/DeepLearning/tiny-dnn/include \
               /home/clark/Compile/cereal-master/include \

LIBS += -lpthread
