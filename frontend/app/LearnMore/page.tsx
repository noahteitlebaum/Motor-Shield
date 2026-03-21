"use client";

import React from "react";
import { Card, CardBody, CardHeader } from "@heroui/card";
import { Chip } from "@heroui/chip";
import { Accordion, AccordionItem } from "@heroui/accordion";

import FadeInUp from "../components/animations/FadeInUp";

export default function LearnMore() {
  return (
    <>
      <style>{`.gradient-background { display: none !important; }`}</style>
      <div className="w-full flex flex-col items-center gap-12 py-12 px-4 pb-24">
        {/* Header */}
        <div className="max-w-4xl w-full flex flex-col gap-4 text-center items-center">
          <FadeInUp delay={0.2}>
            <h1 className="text-5xl font-black tracking-tighter bg-gradient-to-r from-blue-500 to-emerald-500 bg-clip-text text-transparent pb-2">
              How the AI Works
            </h1>
          </FadeInUp>
        </div>

        {/* 1. Pipeline Overview */}
        <div className="max-w-4xl w-full flex flex-col gap-6">
          <FadeInUp delay={0.3}>
            <div className="flex items-center gap-3 mb-2">
              <div className="w-8 h-8 rounded-full bg-blue-500/10 flex items-center justify-center text-blue-500 font-bold">
                1
              </div>
              <h2 className="text-3xl font-bold">Pipeline Overview</h2>
            </div>
            <p className="text-default-500">
              The preprocessing pipeline processes raw CSV telemetry data into
              clean, overlapping windows for the neural networks.
            </p>
          </FadeInUp>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <FadeInUp delay={0.4}>
              <Card className="p-2 bg-default-50 h-full">
                <CardHeader className="pb-0 pt-2 px-4 flex-col items-start">
                  <p className="text-sm font-bold text-blue-500 uppercase">
                    Input Channels
                  </p>
                  <h4 className="font-bold text-large">6 Features</h4>
                </CardHeader>
                <CardBody className="overflow-visible py-2">
                  <p className="text-default-500 text-sm">
                    We capture 3 phase currents (Ia, Ib, Ic) directly, and
                    derive 3 phase voltages (va, vb, vc) from the duty cycles
                    and DC bus voltage.
                  </p>
                </CardBody>
              </Card>
            </FadeInUp>

            <FadeInUp delay={0.5}>
              <Card className="p-2 bg-default-50 h-full">
                <CardHeader className="pb-0 pt-2 px-4 flex-col items-start">
                  <p className="text-sm font-bold text-emerald-500 uppercase">
                    Windowing
                  </p>
                  <h4 className="font-bold text-large">200 Samples</h4>
                </CardHeader>
                <CardBody className="overflow-visible py-2">
                  <p className="text-default-500 text-sm">
                    Signals are sampled at 5,000 Hz. We extract 40ms windows
                    (200 samples) with a 50% overlap (stride of 100) to ensure
                    continuous monitoring.
                  </p>
                </CardBody>
              </Card>
            </FadeInUp>

            <FadeInUp delay={0.6}>
              <Card className="p-2 bg-default-50 h-full">
                <CardHeader className="pb-0 pt-2 px-4 flex-col items-start">
                  <p className="text-sm font-bold text-purple-500 uppercase">
                    Labeling
                  </p>
                  <h4 className="font-bold text-large">Fault Detection</h4>
                </CardHeader>
                <CardBody className="overflow-visible py-2">
                  <p className="text-default-500 text-sm">
                    Windows before the fault timestamp are strictly labeled
                    &quot;Healthy&quot;. Windows after are labeled with the specific fault
                    type. Cross-boundary windows are discarded.
                  </p>
                </CardBody>
              </Card>
            </FadeInUp>
          </div>
        </div>

        {/* 2. Data Augmentation */}
        <div className="max-w-4xl w-full flex flex-col gap-6 mt-8">
          <FadeInUp>
            <div className="flex items-center gap-3 mb-2">
              <div className="w-8 h-8 rounded-full bg-emerald-500/10 flex items-center justify-center text-emerald-500 font-bold">
                2
              </div>
              <h2 className="text-3xl font-bold">Data Augmentation</h2>
            </div>
            <p className="text-default-500 mb-4">
              To make our models robust to real-world hardware variance, we
              augment the training data with hardware-realistic noise. We
              strictly isolate augmentations by split to prevent data leakage.
            </p>
          </FadeInUp>

          <FadeInUp>
            <Card className="bg-default-50 border border-default-200">
              <CardBody className="p-6 md:p-8 flex flex-col md:flex-row gap-8">
                <div className="flex-1 space-y-4">
                  <h3 className="text-xl font-bold">Techniques</h3>
                  <ul className="space-y-3 text-sm text-default-600">
                    <li className="flex gap-2">
                      <span className="text-emerald-500 font-bold">•</span>
                      <span>
                        <strong>Base Augmentations:</strong> Gaussian noise
                        (3-5%), gain variation, and linear offset drift.
                      </span>
                    </li>
                    <li className="flex gap-2">
                      <span className="text-emerald-500 font-bold">•</span>
                      <span>
                        <strong>Advanced Probability:</strong> Random transient
                        spikes, 3rd/5th harmonic distortions, and slight time
                        warping.
                      </span>
                    </li>
                  </ul>
                </div>
                <div className="flex-1 space-y-4">
                  <h3 className="text-xl font-bold">Split Strategy</h3>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center bg-background p-3 rounded-lg border border-default-100">
                      <span className="font-medium">Training Set</span>
                      <Chip color="primary" variant="flat">
                        10x Augmented
                      </Chip>
                    </div>
                    <div className="flex justify-between items-center bg-background p-3 rounded-lg border border-default-100">
                      <span className="font-medium">Validation Set</span>
                      <Chip color="warning" variant="flat">
                        2x Light Noise
                      </Chip>
                    </div>
                    <div className="flex justify-between items-center bg-background p-3 rounded-lg border border-default-100 border-l-4 border-l-success">
                      <span className="font-medium">Test Set</span>
                      <Chip color="success" variant="flat">
                        0x (Original)
                      </Chip>
                    </div>
                  </div>
                </div>
              </CardBody>
            </Card>
          </FadeInUp>
        </div>

        {/* 3. Model Architectures */}
        <div className="max-w-4xl w-full flex flex-col gap-6 mt-8">
          <FadeInUp>
            <div className="flex items-center gap-3 mb-2">
              <div className="w-8 h-8 rounded-full bg-purple-500/10 flex items-center justify-center text-purple-500 font-bold">
                3
              </div>
              <h2 className="text-3xl font-bold">Model Architectures</h2>
            </div>
            <p className="text-default-500">
              MotorShield supports three deep learning models, balancing
              computational efficiency with long-range receptive fields.
            </p>
          </FadeInUp>

          <div className="grid grid-cols-1 gap-6">
            <FadeInUp>
              <Card className="border-l-4 border-l-purple-500 bg-purple-50/30 dark:bg-purple-900/10">
                <CardHeader className="flex gap-3 px-6 pt-6 items-center justify-between">
                  <div className="flex flex-col">
                    <p className="text-md font-bold text-purple-600 dark:text-purple-400">
                      Hybrid CNN–Transformer
                    </p>
                    <p className="text-small text-default-500">
                      ~464k Parameters
                    </p>
                  </div>
                  <Chip color="secondary" size="sm" variant="shadow">
                    Final
                  </Chip>
                </CardHeader>
                <CardBody className="px-6 pb-6 pt-2">
                  <p className="text-default-700 mb-4">
                    The ultimate architecture. A CNN stem rapidly downsamples
                    the 200-sample window into 50 rich feature tokens, which are
                    then processed by a 3-layer Transformer Encoder.
                  </p>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-center text-sm">
                    <div className="bg-background rounded p-2 border border-default-200">
                      <span className="block font-bold">CNN Stem</span>
                      <span className="text-default-500 text-xs text-balance">
                        1D Conv k=7/5
                      </span>
                    </div>
                    <div className="bg-background rounded p-2 border border-default-200">
                      <span className="block font-bold">Attention</span>
                      <span className="text-default-500 text-xs text-balance">
                        3 Layers, 4 Heads
                      </span>
                    </div>
                    <div className="bg-background rounded p-2 border border-default-200">
                      <span className="block font-bold">Pooling</span>
                      <span className="text-default-500 text-xs text-balance">
                        Concat(Avg, Max)
                      </span>
                    </div>
                    <div className="bg-background rounded p-2 border border-default-200">
                      <span className="block font-bold">Output</span>
                      <span className="text-default-500 text-xs text-balance">
                        4 Classes
                      </span>
                    </div>
                  </div>
                </CardBody>
              </Card>
            </FadeInUp>

            <FadeInUp>
              <Card className="border-l-4 border-l-blue-500">
                <CardHeader className="flex gap-3 px-6 pt-6">
                  <div className="flex flex-col">
                    <p className="text-md font-bold">Improved CNN</p>
                    <p className="text-small text-default-500">
                      ~2.1M Parameters
                    </p>
                  </div>
                </CardHeader>
                <CardBody className="px-6 pb-6 pt-2">
                  <p className="text-default-600">
                    Uses deep residual connections (1D Convs) and channel
                    attention to capture complex local patterns in multi-channel
                    time-series data. It excels at extracting local feature
                    signatures like sudden spikes or phase shifts.
                  </p>
                </CardBody>
              </Card>
            </FadeInUp>

            <FadeInUp>
              <Card className="border-l-4 border-l-emerald-500">
                <CardHeader className="flex gap-3 px-6 pt-6">
                  <div className="flex flex-col">
                    <p className="text-md font-bold">Transformer</p>
                    <p className="text-small text-default-500">
                      ~0.4M Parameters
                    </p>
                  </div>
                </CardHeader>
                <CardBody className="px-6 pb-6 pt-2">
                  <p className="text-default-600">
                    Applies multi-head self-attention directly across the time
                    dimension. While slower on long sequences, it inherently
                    models long-range dependencies and complex, non-periodic
                    transients better than convolutions.
                  </p>
                </CardBody>
              </Card>
            </FadeInUp>
          </div>
        </div>

        {/* 4. Complete Walkthrough */}
        <div className="max-w-4xl w-full flex flex-col gap-6 mt-8">
          <FadeInUp>
            <div className="flex items-center gap-3 mb-2">
              <div className="w-8 h-8 rounded-full bg-warning/10 flex items-center justify-center text-warning font-bold">
                4
              </div>
              <h2 className="text-3xl font-bold">Trace a Prediction</h2>
            </div>
            <p className="text-default-500">
              Here&apos;s exactly what happens to a 40ms slice of motor data when it
              hits the Hybrid model.
            </p>
          </FadeInUp>

          <FadeInUp>
            <Accordion variant="splitted">
              <AccordionItem
                key="1"
                aria-label="1. Input Prep"
                title={<span className="font-bold">1. Scale & Transpose</span>}
              >
                The 200x6 window is immediately scaled using a StandardScaler
                strictly fitted on training data, then transposed for the CNN
                channels.
              </AccordionItem>
              <AccordionItem
                key="2"
                aria-label="2. CNN Stem"
                title={
                  <span className="font-bold">2. CNN Feature Extraction</span>
                }
              >
                Two strided 1D convolutions extract local high-frequency
                patterns. This beautifully reduces the sequence length from 200
                to 50, expanding feature channels to 128, cutting attention
                compute costs by 16x.
              </AccordionItem>
              <AccordionItem
                key="3"
                aria-label="3. Tokenization & Positional Encoding"
                title={
                  <span className="font-bold">
                    3. Tokenization & Positional Encoding
                  </span>
                }
              >
                The (50, 128) data is treated as 50 distinct &quot;tokens&quot; across
                time. Sinusoidal positional encoding is added (with 15% dropout)
                to ensure the network understands the sequence order.
              </AccordionItem>
              <AccordionItem
                key="4"
                aria-label="4. Transformer Encoder"
                title={
                  <span className="font-bold">
                    4. Transformer Encoder (3 Layers)
                  </span>
                }
              >
                Multi-head self attention (4 heads) allows any timestep
                constraint to relate to any other in the 50-token sequence. A
                feedforward network (128→256→128) refines the token features.
              </AccordionItem>
              <AccordionItem
                key="5"
                aria-label="5. Dual Pooling"
                title={
                  <span className="font-bold">5. Dual Global Pooling</span>
                }
              >
                The network computes the Global Average Pool and Global Max Pool
                across time and concatenates them into a 256-D vector, capturing
                both the dominant spikes and subtle continuous signals.
              </AccordionItem>
              <AccordionItem
                key="6"
                aria-label="6. Classifier"
                title={<span className="font-bold">6. Classifier Head</span>}
              >
                A Linear layer shrinks the 256-D vector to 128 (with GELU and
                25% dropout), and a final Linear layer collapses it to 4 logits.
                Softmax yields the final class probabilities.
              </AccordionItem>
            </Accordion>
          </FadeInUp>
        </div>
      </div>
    </>
  );
}
