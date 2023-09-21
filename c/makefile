progs = naive_matmul cache_matmul threaded_matmul
src = $(progs:=.c)
headers = stopwatch.h

CC = clang
FLAGS = -Wall -Wextra -Werror -O3
LINKS = -lpthread -lm

all: $(progs) 
clean:
	rm -f $(progs) *.o


%: %.c $(headers)
	$(CC) $(FLAGS) $(LINKS) -o $@ $<