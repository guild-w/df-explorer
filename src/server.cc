#include <stdio.h>
#include <time.h>
#include <iostream>
#include <csignal>
#include <time.h>

#include "workflow/WFHttpServer.h"
#include "workflow/WFFacilities.h"
#include "nlohmann/json.hpp"
#include "darkforest.h"

using json = nlohmann::json;


std::string parse_body(WFHttpTask *server_task) {
    const void *body;
    size_t body_len;
    server_task->get_req()->get_parsed_body(&body, &body_len);
    return std::string((char *) body, body_len);
}

void cors_allow_any(WFHttpTask *server_task)
{
    if (strcmp("OPTIONS", server_task->get_req()->get_method()) == 0) {
        server_task->get_resp()->add_header_pair("Access-Control-Allow-Headers", "content-type");
        server_task->get_resp()->add_header_pair("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    }
    server_task->get_resp()->add_header_pair("Access-Control-Allow-Origin", "*");
}

std::string get_time()
{
    time_t t = time(nullptr);
    char t_str[64];
    strftime(t_str, sizeof(t_str), "%Y-%m-%d %H:%M:%S", localtime(&t));
    return std::string(t_str);
}

void process(WFHttpTask *server_task) {
    std::string method(server_task->get_req()->get_method());
    std::string uri(server_task->get_req()->get_request_uri());

    if (uri != "/explore" ) {
        return;
    }

    cors_allow_any(server_task);
    if (method == "POST") {
        auto body =parse_body(server_task);
        try {
            darkforest::ExploreTask task = json::parse(parse_body(server_task));
            darkforest::ExploreResult result;
            darkforest::explore_chunk(task, result);
            json resp = result;
            server_task->get_resp()->append_output_body(resp.dump());
            server_task->get_resp()->add_header_pair("Content-Type", "application/json");
        } catch (...) {
            printf("[%s] can not process task: %s\n",get_time().c_str(), body.c_str());
            server_task->get_resp()->set_status_code("500");
            server_task->get_resp()->append_output_body("<html>500 Internal Server Error.</html>");
        }
    }
}

static WFFacilities::WaitGroup wait_group(1);

void sig_handler(int signo) {
    wait_group.done();
}

int main()
{
    init_device();

    struct WFServerParams params = HTTP_SERVER_PARAMS_DEFAULT;
    params.request_size_limit = 16 * 1024;

    WFHttpServer server(process);

    signal(SIGINT, sig_handler);

    const int port = 8880;
    if (server.start(port) == 0) {  // start server on port 8888
        printf("start server on port %d\n", port);
        getchar(); // press "Enter" to end.
        server.stop();
    } else {
        perror("Cannot start server\n");
        exit(1);
    }

    return 0;
}
