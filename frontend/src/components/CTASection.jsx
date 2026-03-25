import React from 'react';
import { motion } from 'framer-motion';
import { FiArrowRight } from 'react-icons/fi';

const CTASection = () => {
  return (
    <section className="py-24 bg-white flex justify-center px-6">
      <motion.div 
        initial={{ opacity: 0, y: 30 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="w-full max-w-5xl bg-[#141414] rounded-[3rem] p-12 md:p-20 text-center relative overflow-hidden shadow-2xl"
      >
        {/* Decorative background circle */}
        <div className="absolute top-0 right-0 w-64 h-64 bg-green-500/10 rounded-full translate-x-1/2 -translate-y-1/2 blur-3xl" />
        <div className="absolute bottom-0 left-0 w-64 h-64 bg-green-600/5 rounded-full -translate-x-1/2 translate-y-1/2 blur-3xl" />

        <div className="relative z-10">
          <h2 className="text-3xl md:text-5xl font-black text-white tracking-tight mb-6">
            Ready to <span className="text-green-500 italic">analyze</span> your rooftop?
          </h2>
          <p className="text-gray-400 text-lg md:text-xl max-w-2xl mx-auto mb-10 font-medium">
            Join thousands of homeowners and companies making better energy decisions with SolarScan.
          </p>
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <button 
              onClick={() => document.getElementById('analyze')?.scrollIntoView({ behavior: 'smooth' })}
              className="w-full sm:w-auto px-10 py-5 bg-[#16a34a] text-white rounded-full font-bold text-lg hover:bg-[#15803d] transition-all flex items-center justify-center gap-3 shadow-xl shadow-green-600/20"
            >
              Start Analyzing Now <FiArrowRight />
            </button>
            <button className="w-full sm:w-auto px-10 py-5 bg-white/5 text-white border border-white/10 rounded-full font-bold text-lg hover:bg-white/10 transition-all">
              Contact Sales
            </button>
          </div>
        </div>
      </motion.div>
    </section>
  );
};

export default CTASection;
