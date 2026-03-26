import React from 'react';
import { FiGithub, FiTwitter, FiLinkedin, FiMail } from 'react-icons/fi';

const Footer = () => {
  return (
    <footer className="py-12 bg-[#f8faf8] border-t border-gray-100 px-6">
      <div className="max-w-7xl mx-auto">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-12 mb-12">
          <div className="col-span-1 md:col-span-1">
            <div className="flex items-center gap-2 mb-6">
              <div className="w-8 h-8 bg-[#16a34a] rounded-lg flex items-center justify-center text-white">
                <div className="w-2 h-2 bg-white rounded-full" />
              </div>
              <span className="text-xl font-black text-gray-900 tracking-tighter">SolarScan</span>
            </div>
            <p className="text-gray-400 text-sm font-medium leading-relaxed">
              Advancing the solar revolution through precision computer vision and satellite intelligence.
            </p>
          </div>

          <div>
            <h4 className="text-xs font-black text-gray-900 uppercase tracking-widest mb-6 border-b border-gray-100 pb-2">Product</h4>
            <ul className="space-y-4">
              <li><a href="#" className="text-sm font-medium text-gray-400 hover:text-green-600 transition-colors">Detection API</a></li>
              <li><a href="#" className="text-sm font-medium text-gray-400 hover:text-green-600 transition-colors">Map Integration</a></li>
              <li><a href="#" className="text-sm font-medium text-gray-400 hover:text-green-600 transition-colors">Pricing</a></li>
            </ul>
          </div>

          <div>
            <h4 className="text-xs font-black text-gray-900 uppercase tracking-widest mb-6 border-b border-gray-100 pb-2">Support</h4>
            <ul className="space-y-4">
              <li><a href="#" className="text-sm font-medium text-gray-400 hover:text-green-600 transition-colors">Documentation</a></li>
              <li><a href="#" className="text-sm font-medium text-gray-400 hover:text-green-600 transition-colors">API Status</a></li>
              <li><a href="#" className="text-sm font-medium text-gray-400 hover:text-green-600 transition-colors">Help Center</a></li>
            </ul>
          </div>

          <div>
            <h4 className="text-xs font-black text-gray-900 uppercase tracking-widest mb-6 border-b border-gray-100 pb-2">Connect</h4>
            <div className="flex gap-4">
              <a 
                href="https://github.com/Jrdheeraj" 
                target="_blank" 
                rel="noopener noreferrer"
                className="w-10 h-10 bg-white border border-gray-100 rounded-xl flex items-center justify-center text-gray-400 hover:text-green-600 hover:border-green-200 transition-all shadow-sm"
              >
                <FiGithub />
              </a>
              <a 
                href="https://www.linkedin.com/in/kannemadugu-dheeraj-479515289/" 
                target="_blank" 
                rel="noopener noreferrer"
                className="w-10 h-10 bg-white border border-gray-100 rounded-xl flex items-center justify-center text-gray-400 hover:text-green-600 hover:border-green-200 transition-all shadow-sm"
              >
                <FiLinkedin />
              </a>
              <a 
                href="mailto:jrdheeraj5@gmail.com" 
                className="w-10 h-10 bg-white border border-gray-100 rounded-xl flex items-center justify-center text-gray-400 hover:text-green-600 hover:border-green-200 transition-all shadow-sm"
              >
                <FiMail />
              </a>
            </div>
          </div>
        </div>

        {/* COPYRIGHT AND BOTTOM LINKS REMOVED AS REQUESTED */}
      </div>
    </footer>
  );
};

export default Footer;
