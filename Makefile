CC = cc
COMPILER = gcc
FLAGS = -O3 -flto -funroll-loops -march=native -ftree-vectorize -Iinc -fPIC -Wall -fopenmp -mavx
# -fno-math-errno -ffast-math -fomit-frame-pointer -ftree-parallelize-loops -fprofile-generate -fprofile-use
LIBRARIES = -lpng -ljpeg -lcjson `sdl2-config --cflags` `sdl2-config --libs` -lm
RM = rm -rf
MKDIR = mkdir -p

STATIC_LIB = libican.a
SHARED_LIB = libican.so

SRC_DIR = src
INC_DIR = inc
BUILD_DIR = build
OBJS_DIR = $(BUILD_DIR)/objs
OUT_DIR = $(BUILD_DIR)/out

SRCS = $(wildcard $(SRC_DIR)/*.c)
INCS = $(wildcard $(INC_DIR)/*.h)
OBJS = $(SRCS:$(SRC_DIR)/%.c=$(OBJS_DIR)/%.o)

PREFIX = /usr/local
INCLUDE_DIR = $(PREFIX)/include
LIB_DIR = $(PREFIX)/lib

all: static shared

static: $(OBJS)
	ar rcs $(OUT_DIR)/$(STATIC_LIB) $(OBJS)

shared: $(OBJS)
	$(CC) -shared -o $(OUT_DIR)/$(SHARED_LIB) $(OBJS)

$(OBJS_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(FLAGS) -c $< -o $@

$(BUILD_DIR):
	$(MKDIR) $(BUILD_DIR)
	$(MKDIR) $(OBJS_DIR)
	$(MKDIR) $(OUT_DIR)

install: static shared
	$(MKDIR) $(INCLUDE_DIR)
	cp $(INC_DIR)/*.h $(INCLUDE_DIR)

	$(MKDIR) $(LIB_DIR)
	cp $(OUT_DIR)/$(STATIC_LIB) $(LIB_DIR)
	cp $(OUT_DIR)/$(SHARED_LIB) $(LIB_DIR)

	ldconfig

run: clean static
	$(COMPILER) $(FLAGS) main.c $(LIBRARIES) -L$(OUT_DIR) -lican && ./a.out

clean:
	$(RM) $(OUT_DIR)/*

fclean:
	$(RM) $(BUILD_DIR) $(STATIC_LIB) $(SHARED_LIB)

gtest:
	$(COMPILER) -g main.c $(FLAGS) $(LIBRARIES) -L$(OUT_DIR) -lican && gdb ./a.out

memcheck: run
	valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all ./a.out

.PHONY: all static shared install run clean fclean gtest memcheck