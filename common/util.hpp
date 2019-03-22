#ifndef UTIL_H //二重でincludeされることを防ぐ
#define UTIL_H

#include <vector>

using namespace std;

template <class T>
void cout_vector(T vec)
{
    cout << "{ ";
    for (int i = 0; i < vec.size(); i++)
    {
        if (i > 0)
        {
            cout << ", ";
        }
        cout << vec.at(i);
    }
    cout << " }" << endl;
}

#endif