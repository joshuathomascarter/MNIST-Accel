//      // verilator_coverage annotation
        // DESCRIPTION: Verilator: built-in packages and classes
        //
        // Code available from: https://verilator.org
        //
        //*************************************************************************
        //
        // Copyright 2022-2025 by Wilson Snyder. This program is free software; you can
        // redistribute it and/or modify it under the terms of either the GNU Lesser
        // General Public License Version 3 or the Perl Artistic License Version 2.0.
        // SPDX-License-Identifier: LGPL-3.0-only OR Artistic-2.0
        //
        //*************************************************************************
        ///
        /// \file
        /// \brief Verilated IEEE std:: header
        ///
        /// This file is included automatically by Verilator, unless '--no-std-package'
        /// is used.
        ///
        /// This file is not part of the Verilated public-facing API.
        /// It is only for internal use.
        ///
        //*************************************************************************
        //
        // The following keywords from this file are hardcoded for detection in the parser:
        // "mailbox", "process", "randomize", "semaphore", "std"
        
        // verilator lint_off DECLFILENAME
        // verilator lint_off TIMESCALEMOD
        // verilator lint_off UNUSEDSIGNAL
        package std;
          class mailbox #(
              type T
          );
            protected int m_bound;
            protected T m_queue[$];
        
            function new(int bound = 0);
              m_bound = bound;
            endfunction
        
            function int num();
              return m_queue.size();
            endfunction
        
            task put(T message);
        `ifdef VERILATOR_TIMING
              while (m_bound != 0 && m_queue.size() >= m_bound)  //
                wait (m_queue.size() < m_bound);
              m_queue.push_back(message);
        `endif
            endtask
        
            function int try_put(T message);
              if (m_bound == 0 || num() < m_bound) begin
                m_queue.push_back(message);
                return 1;
              end
              return 0;
            endfunction
        
            task get(ref T message);
        `ifdef VERILATOR_TIMING
              while (m_queue.size() == 0) begin
                wait (m_queue.size() > 0);
              end
              message = m_queue.pop_front();
        `endif
            endtask
        
            function int try_get(ref T message);
              if (num() > 0) begin
                message = m_queue.pop_front();
                return 1;
              end
              return 0;
            endfunction
        
            task peek(ref T message);
        `ifdef VERILATOR_TIMING
              while (m_queue.size() == 0) begin
                wait (m_queue.size() > 0);
              end
              message = m_queue[0];
        `endif
            endtask
        
            function int try_peek(ref T message);
              if (num() > 0) begin
                message = m_queue[0];
                return 1;
              end
              return 0;
            endfunction
          endclass
        
          class semaphore;
            protected int m_keyCount;
        
%000000     function new(int keyCount = 0);
%000000       m_keyCount = keyCount;
            endfunction
        
%000000     function void put(int keyCount = 1);
%000000       m_keyCount += keyCount;
            endfunction
        
%000000     task get(int keyCount = 1);
        `ifdef VERILATOR_TIMING
%000000       while (m_keyCount < keyCount) begin
%000000         wait (m_keyCount >= keyCount);
              end
%000000       m_keyCount -= keyCount;
        `endif
            endtask
        
%000000     function int try_get(int keyCount = 1);
%000000       if (m_keyCount >= keyCount) begin
%000000         m_keyCount -= keyCount;
%000000         return 1;
              end
%000000       return 0;
            endfunction
          endclass
        
%000000   class process;
            typedef enum {
              FINISHED = 0,
              RUNNING = 1,
              WAITING = 2,
              SUSPENDED = 3,
              KILLED = 4
            } state;
        
            // Width visitor changes it to VlProcessRef
            protected chandle m_process;
        
%000000     static function process self();
%000000       process p = new;
        `ifdef VERILATOR_TIMING
%000000       $c(p.m_process, " = vlProcess;");
        `endif
%000000       return p;
            endfunction
        
%000000     protected function void set_status(state s);
        `ifdef VERILATOR_TIMING
%000000       $c(m_process, "->state(", s, ");");
        `endif
            endfunction
        
%000000     function state status();
        `ifdef VERILATOR_TIMING
%000000       return state'($cpure(m_process, "->state()"));
        `else
              return RUNNING;
        `endif
            endfunction
        
%000000     function void kill();
%000000       set_status(KILLED);
            endfunction
        
            function void suspend();
              $error("std::process::suspend() not supported");
            endfunction
        
%000000     function void resume();
%000000       set_status(RUNNING);
            endfunction
        
%000000     task await();
        `ifdef VERILATOR_TIMING
%000000       wait (status() == FINISHED || status() == KILLED);
        `endif
            endtask
        
%000000     static task killQueue(ref process processQueue[$]);
        `ifdef VERILATOR_TIMING
%000000       while (processQueue.size() > 0) begin
%000000         processQueue.pop_back().kill();
              end
        `endif
            endtask
        
            // Two process references are equal if the different classes' containing
            // m_process are equal. Can't yet use <=> as the base class template
            // comparisons doesn't define <=> as they don't yet require --timing and C++20.
            // verilog_format: off
        `ifdef VERILATOR_TIMING
        `systemc_header_post
        template<> template<>
        inline bool VlClassRef<`systemc_class_name>::operator==(const VlClassRef<`systemc_class_name>& rhs) const {
            if (!m_objp && !rhs.m_objp) return true;
            if (!m_objp || !rhs.m_objp) return false;
            return m_objp->__PVT__m_process == rhs.m_objp->__PVT__m_process;
        };
        template<> template<>
        inline bool VlClassRef<`systemc_class_name>::operator!=(const VlClassRef<`systemc_class_name>& rhs) const {
            if (!m_objp && !rhs.m_objp) return false;
            if (!m_objp || !rhs.m_objp) return true;
            return m_objp->__PVT__m_process != rhs.m_objp->__PVT__m_process;
        };
        template<> template<>
        inline bool VlClassRef<`systemc_class_name>::operator<(const VlClassRef<`systemc_class_name>& rhs) const {
            if (!m_objp && !rhs.m_objp) return false;
            if (!m_objp || !rhs.m_objp) return false;
            return m_objp->__PVT__m_process < rhs.m_objp->__PVT__m_process;
        };
        `verilog
        `endif
            // verilog_format: on
        
            // When really implemented, srandom must operate on the process, but for
            // now rely on the srandom() that is automatically generated for all
            // classes.
            //
            // function void srandom(int seed);
            // endfunction
        
            // The methods below access the common RNG, full support
            // of get_randstate/set_randstate requires accessing the RNG state
            // of the specified process (see IEEE 1800-2023, 18.14.), but as for
            // now processes do not have their own RNGs.
%000000     function string get_randstate();
              // Initialize with $c to ensure it won't be constified
%000000       string s = string'($c("0"));
        
%000000       $c(s, " = ", m_process, "->randstate();");
%000000       return s;
            endfunction
        
%000000     function void set_randstate(string s);
%000000       $c(m_process, "->randstate(", s, ");");
            endfunction
          endclass
        
%000000   function int randomize();
%000000     randomize = 0;
          endfunction
        endpackage
        
