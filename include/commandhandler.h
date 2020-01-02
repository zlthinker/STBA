#ifndef COMMANDHANDLER_H
#define COMMANDHANDLER_H

#include <string>
#include <vector>

#if defined(__cplusplus)
extern "C" {
#endif

bool commandHandler(std::vector<std::string> & commandOptions);
void commandHelp();

#if defined(__cplusplus)
}
#endif

#endif // COMMANDHANDLER_H
