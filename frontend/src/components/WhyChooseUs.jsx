import React from "react";
import { motion } from "framer-motion";
import { FiZap, FiCpu, FiMaximize, FiShield } from "react-icons/fi";

const WhyChooseUs = () => {
  return (
    <section className="w-full py-24 text-white bg-[hsl(152,60%,32%)] overflow-hidden">
      <div className="max-w-7xl mx-auto px-6">
        <div className="mb-12 flex flex-col items-start justify-center text-center px-8 text-white w-full">
          <h2 className="max-w-4xl text-4xl font-black md:text-5xl leading-tight text-center mx-auto">
            Scale your solar strategy with our <br className="hidden md:block" />
            <span className="text-green-100/70">all-in-one AI platform</span>
          </h2>
        </div>

        <div className="mb-4 grid grid-cols-12 gap-4">
          <BounceCard className="col-span-12 md:col-span-4">
            <CardHeader icon={<FiZap />} title="High Accuracy" />
            <p className="text-slate-500 mt-4 leading-relaxed font-medium">
              Our YOLO-based AI models are trained on diverse satellite datasets specifically optimized for rooftop structures.
            </p>
          </BounceCard>

          <BounceCard className="col-span-12 md:col-span-8">
            <CardHeader icon={<FiCpu />} title="Real-time Processing" />
            <p className="text-slate-500 mt-4 leading-relaxed font-medium max-w-md">
              Get results in under 5 seconds with our high-performance inference engine running on edge GPU clusters globally.
            </p>
          </BounceCard>
        </div>

        <div className="grid grid-cols-12 gap-4">
          <BounceCard className="col-span-12 md:col-span-8">
            <CardHeader icon={<FiMaximize />} title="Precise Area Estimation" />
            <p className="text-slate-500 mt-4 leading-relaxed font-medium max-w-md">
              Advanced segmentation algorithms calculate solar panel area with 99.4% precision for accurate ROI and energy generation calculation.
            </p>
          </BounceCard>

          <BounceCard className="col-span-12 md:col-span-4">
            <CardHeader icon={<FiShield />} title="Scalable Architecture" />
            <p className="text-slate-500 mt-4 leading-relaxed font-medium">
              Designed to handle millions of requests, making it perfect for smart city planning and large-scale assessments.
            </p>
          </BounceCard>
        </div>
      </div>
    </section>
  );
};

const BounceCard = ({ className, children }) => {
  return (
    <motion.div
      whileHover={{ scale: 0.98, rotate: "-0.5deg" }}
      className={`group relative h-full cursor-pointer overflow-hidden rounded-[2.5rem] bg-slate-50 p-8 border border-slate-100 transition-colors hover:bg-white ${className}`}
    >
      {children}
    </motion.div>
  );
};

const CardHeader = ({ icon, title }) => {
  return (
    <div className="flex items-center gap-4">
      <div className="w-12 h-12 rounded-2xl bg-white shadow-sm flex items-center justify-center text-2xl text-slate-900 border border-slate-100">
        {icon}
      </div>
      <h3 className="text-2xl font-bold text-slate-800">{title}</h3>
    </div>
  );
};

export default WhyChooseUs;
