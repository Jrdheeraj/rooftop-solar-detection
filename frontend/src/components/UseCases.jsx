import React from 'react';
import { motion } from 'framer-motion';
import { FiHome, FiTrendingUp, FiSettings, FiBriefcase } from 'react-icons/fi';

const cases = [
  {
    icon: <FiHome className="text-2xl" />,
    title: "Homeowners",
    desc: "Calculate your home's solar potential and estimated savings before talking to an installer."
  },
  {
    icon: <FiTrendingUp className="text-2xl" />,
    title: "Energy Companies",
    desc: "Scan entire neighborhoods to identify lead generation opportunities and manage grid load."
  },
  {
    icon: <FiSettings className="text-2xl" />,
    title: "Governments",
    desc: "Use high-level analytics for urban planning and monitoring renewable energy adoption."
  },
  {
    icon: <FiBriefcase className="text-2xl" />,
    title: "Real Estate",
    desc: "Add value to property listings by showcasing rooftop energy efficiency and solar capacity."
  }
];

const UseCases = () => {
  return (
    <section className="py-24 bg-[#f8faf8]">
      <div className="max-w-7xl mx-auto px-6">
        <div className="flex flex-col lg:flex-row gap-16 items-center">
          <div className="lg:w-1/3">
            <h2 className="text-3xl font-black text-gray-900 tracking-tight mb-6">
              Empowering <span className="text-[#16a34a]">everyone</span> in the energy transition.
            </h2>
            <p className="text-gray-500 font-medium leading-relaxed mb-8">
              Whether you're a single property owner or a large-scale utility provider, SolarScan provides the precision you need.
            </p>
          </div>
          
          <div className="lg:w-2/3 grid grid-cols-1 sm:grid-cols-2 gap-6">
            {cases.map((c, i) => (
              <motion.div
                key={i}
                whileHover={{ scale: 1.02 }}
                className="p-8 bg-white rounded-[2rem] border border-transparent hover:border-green-100 shadow-sm transition-all"
              >
                <div className="w-12 h-12 bg-green-50 text-green-600 rounded-xl flex items-center justify-center mb-6">
                  {c.icon}
                </div>
                <h3 className="text-lg font-bold text-gray-900 mb-2">{c.title}</h3>
                <p className="text-sm text-gray-500 leading-relaxed font-medium">
                  {c.desc}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default UseCases;
