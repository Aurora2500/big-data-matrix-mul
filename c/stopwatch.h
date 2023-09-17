#ifndef STOPWATCH_H
#define STOPWATCH_H
#include <sys/time.h>
#include <stdlib.h>

struct stopwatch {
		struct timeval start;
		struct timeval stop;
};

static inline void stopwatch_start(struct stopwatch *sw)
{
	gettimeofday(&sw->start, NULL);
}

static inline void stopwatch_stop(struct stopwatch *sw)
{
	gettimeofday(&sw->stop, NULL);
}

static inline double stopwatch_elapsed(struct stopwatch *sw)
{
	return (sw->stop.tv_sec - sw->start.tv_sec) + (sw->stop.tv_usec - sw->start.tv_usec) / 1000000.0;
}

 #endif // STOPWATCH_H