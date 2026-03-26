import React from 'react';
import { motion } from 'framer-motion';
import { FiChevronDown, FiArrowRight, FiZap } from 'react-icons/fi';

const Navbar = () => {
  return (
    <nav className="absolute top-0 left-0 right-0 z-[100] bg-transparent">
      <div className="max-w-7xl mx-auto px-8 h-20 flex items-center justify-between">
        {/* LOGO */}
        <a href="#home" className="flex items-center gap-2.5 group cursor-pointer">
          <div className="w-9 h-9 bg-black rounded-xl flex items-center justify-center text-white shadow-lg group-hover:scale-110 transition-transform duration-300">
            <FiZap className="text-xl" />
          </div>
          <span className="text-xl font-black tracking-tight text-slate-900 font-inter">
            SolarScan
          </span>
        </a>

        {/* NAV LINKS */}
        <div className="hidden md:flex items-center gap-10">
          <NavLink label="Home" href="#home" />
          <NavLink label="How it Works" href="#how-it-works" />
          <NavLink label="Analyze" href="#analyze" />
          <NavLink label="Features" href="#features" />
          <NavLink label="Solutions" href="#solutions" hasDropdown />
          <NavLink label="Contact" href="#about" />
        </div>

        {/* CTA BUTTON */}
        <motion.a
          href="#analyze"
          whileHover={{ scale: 1.05, backgroundColor: '#333' }}
          whileTap={{ scale: 0.95 }}
          className="bg-black text-white px-7 py-3 rounded-full font-bold flex items-center gap-2.5 text-sm transition-all shadow-2xl hover:shadow-black/20"
        >
          Analyze Now
          <FiArrowRight className="text-lg" />
        </motion.a>
      </div>
    </nav>
  );
};

const NavLink = ({ label, href, hasDropdown }) => (
  <a 
    href={href}
    className="flex items-center gap-1.5 text-slate-500 font-semibold hover:text-black transition-all cursor-pointer group text-sm uppercase tracking-wider"
  >
    <span>{label}</span>
    {hasDropdown && (
      <FiChevronDown className="text-slate-400 group-hover:text-black transition-colors" />
    )}
  </a>
);

export default Navbar;
