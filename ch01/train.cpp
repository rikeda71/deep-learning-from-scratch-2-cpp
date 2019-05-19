#include "../common/optimizer.hpp"
#include "../common/trainer.hpp"
#include "../dataset/spiral.hpp"
#include "./two_layer_net.hpp"

int main()
{
    int max_epoch = 300;
    int batch_size = 30;
    int hidden_size = 10;
    double learning_rate = 1.0;

    auto [x, t] = load_data();
    auto model = TwoLayerNet(2, hidden_size, 3);
    auto optimizer = SGD(learning_rate);

    auto trainer = Trainer(&model, &optimizer);
    trainer.fit(x, t, max_epoch, batch_size, 10);
    trainer.plot();
    return 0;
}