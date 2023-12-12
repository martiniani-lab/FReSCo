#ifndef _FRESCO_QUEUE_HPP_
#define _FRESCO_QUEUE_HPP_

#include <mutex>
#include <queue>

/**
 * A queue with a thread-safe push method
 */
template <typename T>
class SafePushQueue : public std::queue<T>
{
protected:
    std::mutex m_mutex;

public:
    void push( const T& item )
    {
        std::unique_lock<std::mutex> mlock(m_mutex);
        std::queue<T>::push(item);
        mlock.unlock();
    }
};

#endif // #ifndef _FRESCO_QUEUE_HPP_
