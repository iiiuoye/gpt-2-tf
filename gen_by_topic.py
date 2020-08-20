#!/usr/bin/env python3
import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder

def pre_load_model():
    model_name='kaggle_news'
    seed=None
    length=1024
    nsamples=10
    batch_size=1
    temperature=0.7
    top_k=40
    top_p=0.9

    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)
    

    sess = tf.Session()
    context = tf.placeholder(tf.int32, [batch_size, None])

    np.random.seed(seed)
    tf.set_random_seed(seed)

    output = sample.sample_sequence(
        hparams=hparams, length=length,
        context=context,
        batch_size=batch_size,
        temperature=temperature, top_k=top_k, top_p=top_p
    )

    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
    saver.restore(sess, ckpt)

    return sess, enc, context, output


def gpt2_gen_text(sess, enc, context, output, prompt_text):
    nsamples=3
    batch_size=1

    raw_text = prompt_text
    while not raw_text:
        print('Prompt should not be empty!')
        raw_text = " "
        return raw_text
    context_tokens = enc.encode(raw_text)
    generated = 0

    gen_list = []
    for _ in range(nsamples // batch_size):
        out = sess.run(output, feed_dict={
            context: [context_tokens for _ in range(batch_size)]
        })[:, len(context_tokens):]
        for i in range(batch_size):
            generated += 1
            text = enc.decode(out[i])
            # print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
            # print(text)
            gen_list.append(text)
    return gen_list
    # print("=" * 80)


if __name__ == '__main__':
    sess, enc, context, output = pre_load_model()
    text = gpt2_gen_text(sess, enc, context, output, "China will lead the world. | ")
    print(len(text))
