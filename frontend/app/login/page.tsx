import React from 'react';
import Link from 'next/link';

export default function LoginPage() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-[var(--background)] p-4 sm:p-8">
      {/* Background decoration */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-1/2 -left-1/2 w-full h-full bg-gradient-to-br from-indigo-500/10 to-purple-500/10 blur-[120px] rounded-full"></div>
        <div className="absolute -bottom-1/2 -right-1/2 w-full h-full bg-gradient-to-tl from-blue-500/10 to-teal-500/10 blur-[120px] rounded-full"></div>
      </div>

      <div className="relative w-full max-w-md bg-zinc-900/80 backdrop-blur-xl rounded-3xl shadow-2xl overflow-hidden border border-zinc-800/50 transition-all hover:border-zinc-700/50">
        <div className="p-8 sm:p-10">
          <div className="text-center mb-8">
            <h1 className="text-3xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-indigo-400 mb-2 tracking-tight">
              Welcome Back
            </h1>
            <p className="text-zinc-400 text-sm font-medium">
              Sign in to your account to continue
            </p>
          </div>
          
          <form className="space-y-6">
            <div className="space-y-2">
              <label className="block text-sm font-semibold text-zinc-300" htmlFor="email">
                Email Address
              </label>
              <div className="relative group">
                <input
                  id="email"
                  type="email"
                  placeholder="you@example.com"
                  className="w-full px-4 py-3.5 bg-zinc-950/50 border border-zinc-800 rounded-xl text-white placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500 transition-all duration-300"
                  required
                />
              </div>
            </div>
            
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="block text-sm font-semibold text-zinc-300" htmlFor="password">
                  Password
                </label>
                <Link href="#" className="text-sm font-medium text-blue-400 hover:text-blue-300 transition-colors">
                  Forgot password?
                </Link>
              </div>
              <div className="relative group">
                <input
                  id="password"
                  type="password"
                  placeholder="••••••••"
                  className="w-full px-4 py-3.5 bg-zinc-950/50 border border-zinc-800 rounded-xl text-white placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500 transition-all duration-300"
                  required
                />
              </div>
            </div>

            <div className="flex items-center pt-2">
              <input
                id="remember_me"
                type="checkbox"
                className="h-4 w-4 bg-zinc-950 border-zinc-800 rounded text-blue-500 focus:ring-blue-500 focus:ring-offset-zinc-900 transition-all cursor-pointer"
              />
              <label htmlFor="remember_me" className="ml-3 block text-sm font-medium text-zinc-300 cursor-pointer select-none">
                Remember me for 30 days
              </label>
            </div>

            <button
              type="submit"
              className="w-full mt-8 py-3.5 px-4 flex justify-center items-center gap-2 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white text-base font-semibold rounded-xl transition-all duration-300 shadow-lg shadow-blue-500/25 hover:shadow-blue-500/40 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-zinc-900 active:scale-[0.98]"
            >
              Sign In
            </button>
          </form>
          
          <div className="mt-8 pt-6 border-t border-zinc-800/50 text-center text-sm text-zinc-400 font-medium">
            Don't have an account?{' '}
            <Link href="/register" className="text-blue-400 hover:text-blue-300 font-semibold transition-colors">
              Sign up
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
