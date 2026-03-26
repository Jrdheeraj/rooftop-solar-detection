import React, { useRef } from 'react';
import { motion, useMotionValue, useSpring, useTransform } from 'framer-motion';
import { FiArrowRight } from 'react-icons/fi';

const CTASection = () => {
  const cardRef = useRef(null);
  
  // Motion values for tilt
  const x = useMotionValue(0);
  const y = useMotionValue(0);

  // Smooth springs for rotation
  const mouseXSpring = useSpring(x);
  const mouseYSpring = useSpring(y);

  // Rotation transformations
  const rotateX = useTransform(mouseYSpring, [-0.5, 0.5], ["10deg", "-10deg"]);
  const rotateY = useTransform(mouseXSpring, [-0.5, 0.5], ["-10deg", "10deg"]);

  // Gradient follow transformations
  const gradientX = useTransform(mouseXSpring, [-0.5, 0.5], ["0%", "100%"]);
  const gradientY = useTransform(mouseYSpring, [-0.5, 0.5], ["0%", "100%"]);

  const handleMouseMove = (e) => {
    if (!cardRef.current) return;
    const rect = cardRef.current.getBoundingClientRect();
    
    // Calculate normalized mouse position (-0.5 to 0.5)
    const width = rect.width;
    const height = rect.height;
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    
    const xPct = mouseX / width - 0.5;
    const yPct = mouseY / height - 0.5;
    
    x.set(xPct);
    y.set(yPct);
  };

  const handleMouseLeave = () => {
    x.set(0);
    y.set(0);
  };

  return (
    <section className="py-24 bg-white flex justify-center px-6 overflow-hidden">
      <motion.div 
        ref={cardRef}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        style={{
          rotateX,
          rotateY,
          transformStyle: "preserve-3d",
        }}
        className="w-full max-w-5xl bg-[#0a0a0a] rounded-[3rem] p-12 md:p-24 text-center relative shadow-[0_20px_50px_rgba(0,0,0,0.5)] border border-white/5 group"
      >
        {/* Animated Solar Aura Background */}
        <motion.div 
          style={{
            background: useTransform(
              [gradientX, gradientY],
              ([gx, gy]) => `radial-gradient(circle at ${gx} ${gy}, rgba(22, 163, 74, 0.15) 0%, transparent 70%)`
            ),
          }}
          className="absolute inset-0 pointer-events-none transition-opacity duration-500 group-hover:opacity-100 opacity-0" 
        />

        {/* Depth Layers */}
        <div style={{ transform: "translateZ(80px)" }} className="relative z-10 transition-transform duration-200">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
          >
             <h2 className="text-4xl md:text-6xl font-black text-white tracking-tight mb-8 leading-tight">
               Ready to <span className="text-[#16a34a] relative">
                 analyze
                 <motion.span 
                   className="absolute -bottom-2 left-0 w-full h-1 bg-[#16a34a]/30 rounded-full"
                   initial={{ width: 0 }}
                   whileInView={{ width: "100%" }}
                   transition={{ delay: 0.5, duration: 1 }}
                 />
               </span> your rooftop?
             </h2>
             
             {/* Interaction Container */}
             <div className="flex justify-center">
               <motion.button 
                 onClick={() => document.getElementById('analyze')?.scrollIntoView({ behavior: 'smooth' })}
                 whileHover={{ 
                   scale: 1.05,
                   boxShadow: "0 0 30px rgba(22, 163, 74, 0.4)"
                 }}
                 whileTap={{ scale: 0.95 }}
                 className="relative px-12 py-6 bg-[#16a34a] text-white rounded-full font-bold text-xl flex items-center gap-4 group transition-colors overflow-hidden"
               >
                 {/* Internal glow effect */}
                 <span className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000" />
                 
                 <span className="relative z-10">Start Analyzing Now</span>
                 <motion.div
                   animate={{ x: [0, 5, 0] }}
                   transition={{ repeat: Infinity, duration: 1.5 }}
                   className="relative z-10"
                 >
                   <FiArrowRight />
                 </motion.div>
               </motion.button>
             </div>
          </motion.div>
        </div>

        {/* Decorative corner elements */}
        <div className="absolute top-10 left-10 w-2 h-2 rounded-full bg-white/10" />
        <div className="absolute top-10 right-10 w-2 h-2 rounded-full bg-white/10" />
        <div className="absolute bottom-10 left-10 w-2 h-2 rounded-full bg-white/10" />
        <div className="absolute bottom-10 right-10 w-2 h-2 rounded-full bg-white/10" />
      </motion.div>
    </section>
  );
};

export default CTASection;
