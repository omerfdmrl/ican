CC = cc
COMPILER = gcc
FLAGS = 
LIBRARIES = -lm -lpng -ljpeg -lcjson `sdl2-config --cflags` `sdl2-config --libs`
RM = rm -f

SOURCE = main.c
OUTPUT = program
NAME = libican.a
TNAME = bin

TPATH = test/
UPATH = test/src/
TDPATH = test/data/
TDSPATH = test/data/small/
OPATH = build/objs/
RPATH = build/results/

CFILES = $(wildcard src/*.c)
OFILES = $(patsubst src/%.c, $(OPATH)%.o, $(CFILES))
TFILES = $(wildcard $(UPATH)*.c)

all: $(NAME)

create_dirs:
	@mkdir -p $(OPATH) $(RPATH) $(TDSPATH)

$(NAME): $(OFILES)
	ar rcs $(RPATH)$(NAME) $^
	ranlib $(RPATH)$(NAME)

$(OPATH)%.o: src/%.c | create_dirs
	$(CC) $(FLAGS) -c $< -o $@

$(TDSPATH):
	@mkdir -p $(TDSPATH)
	$(COMPILER) $(FLAGS) $(TPATH)build.c $(CFILES) $(LIBRARIES) -o $(RPATH)$(TNAME) && ./$(RPATH)$(TNAME)

clean:
	@if [ -f $(OPATH)$(NAME) ]; then $(RM) $(OPATH)$(NAME); fi

fclean: clean
	@if [ -n "$(OFILES)" ]; then $(RM) $(OFILES); fi
	@if [ -f $(TNAME) ]; then $(RM) $(TNAME); fi
	$(RM) -r $(RPATH)*
	$(RM) -r $(OPATH)*
	$(RM) -rf $(TDPATH)

$(OUTPUT): $(NAME)
	$(COMPILER) $(FLAGS) $(SOURCE) $(CFILES) $(LIBRARIES) -o $(RPATH)$(OUTPUT)

run: $(OUTPUT)
	./$(RPATH)$(OUTPUT)

gtest:
	$(COMPILER) -g $(FLAGS) $(SOURCE) $(CFILES) $(LIBRARIES) -o $(RPATH)$(TNAME) && gdb $(RPATH)$(TNAME)

utest: $(TDSPATH)
	$(COMPILER) $(FLAGS) $(TPATH)$(SOURCE) $(CFILES) $(LIBRARIES) $(TFILES) -o $(RPATH)$(TNAME) && ./$(RPATH)$(TNAME)

memcheck: $(OUTPUT)
	valgrind --tool=memcheck --leak-check=full -s ./$(RPATH)$(OUTPUT)

.PHONY: all create_dirs clean fclean run gtest utest memcheck
.DEFAULT_GOAL = run