#include "Process.h"

#include <hyper/platform.h>

#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace hyper
{
#ifdef _WIN32
    namespace
    {
        // Quotes one argument the way CommandLineToArgvW parses it back.
        void AppendQuoted(std::wstring& cmd, const std::wstring& arg)
        {
            cmd += L'"';
            size_t backslashes = 0;
            for (wchar_t c : arg)
            {
                if (c == L'\\')
                {
                    ++backslashes;
                    continue;
                }
                if (c == L'"')
                    cmd.append(backslashes * 2 + 1, L'\\');
                else
                    cmd.append(backslashes, L'\\');
                backslashes = 0;
                cmd += c;
            }
            cmd.append(backslashes * 2, L'\\');
            cmd += L'"';
        }

        std::wstring Widen(const std::string& s)
        {
            if (s.empty())
                return {};
            int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), nullptr, 0);
            std::wstring w(size_t(n), L'\0');
            MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), w.data(), n);
            return w;
        }
    }

    ProcessResult RunProcess(const std::vector<std::string>& args)
    {
        ProcessResult result;
        if (args.empty())
            return result;

        std::wstring cmd;
        for (const auto& a : args)
        {
            if (!cmd.empty())
                cmd += L' ';
            AppendQuoted(cmd, Widen(a));
        }

        SECURITY_ATTRIBUTES sa{ sizeof(sa), nullptr, TRUE };
        HANDLE readPipe = nullptr, writePipe = nullptr;
        if (!CreatePipe(&readPipe, &writePipe, &sa, 0))
            return result;
        SetHandleInformation(readPipe, HANDLE_FLAG_INHERIT, 0);

        STARTUPINFOW si{};
        si.cb = sizeof(si);
        si.dwFlags = STARTF_USESTDHANDLES;
        si.hStdOutput = writePipe;
        si.hStdError = writePipe;
        si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);

        PROCESS_INFORMATION pi{};
        BOOL started = CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW,
                                      nullptr, nullptr, &si, &pi);
        CloseHandle(writePipe);
        if (!started)
        {
            CloseHandle(readPipe);
            result.output = "failed to start " + args[0];
            return result;
        }

        char buf[4096];
        DWORD read = 0;
        while (ReadFile(readPipe, buf, sizeof(buf), &read, nullptr) && read > 0)
            result.output.append(buf, read);
        CloseHandle(readPipe);

        WaitForSingleObject(pi.hProcess, INFINITE);
        DWORD code = 0;
        GetExitCodeProcess(pi.hProcess, &code);
        result.exitCode = int(code);
        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
        return result;
    }
#else
    ProcessResult RunProcess(const std::vector<std::string>& args)
    {
        ProcessResult result;
        if (args.empty())
            return result;

        int fds[2];
        if (pipe(fds) != 0)
            return result;

        pid_t pid = fork();
        if (pid == 0)
        {
            dup2(fds[1], STDOUT_FILENO);
            dup2(fds[1], STDERR_FILENO);
            close(fds[0]);
            close(fds[1]);
            std::vector<char*> argv;
            for (const auto& a : args)
                argv.push_back(const_cast<char*>(a.c_str()));
            argv.push_back(nullptr);
            execv(argv[0], argv.data());
            _exit(127);
        }
        close(fds[1]);

        char buf[4096];
        ssize_t n;
        while ((n = read(fds[0], buf, sizeof(buf))) > 0)
            result.output.append(buf, size_t(n));
        close(fds[0]);

        int status = 0;
        waitpid(pid, &status, 0);
        result.exitCode = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
        return result;
    }
#endif
}
