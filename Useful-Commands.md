Sure! Here’s the same list formatted in Markdown for better readability:

```markdown
# Useful PBS Commands for Job Management

20 useful commands for managing jobs on a PBS for CCAST, 

1. **View all available jobs**
   ```bash
   qstat
   ```
   Displays the status of all jobs in the queue.

2. **View your own jobs**
   ```bash
   qstat -u your_username
   ```
   Shows jobs submitted by your user.

3. **View detailed information about a specific job**
   ```bash
   qstat -f job_id
   ```
   Provides detailed information about a specific job.

4. **View jobs in a specific queue**
   ```bash
   qstat -Q queue_name
   ```
   Lists jobs in a specified queue.

5. **Check the status of all nodes**
   ```bash
   pbsnodes -av
   ```
   Shows the status and availability of all compute nodes.

6. **Check resources on nodes**
   ```bash
   pbsnodes -l
   ```
   Lists the resources available on each node.

7. **View the queue of jobs in detail**
   ```bash
   qstat -f
   ```
   Displays detailed information for all jobs in the queue.

8. **Delete a specific job**
   ```bash
   qdel job_id
   ```
   Removes a job from the queue using its job ID.

9. **Check the history of your jobs**
   ```bash
   showq
   ```
   Displays a summary of jobs that have been completed or are currently in the queue.

10. **View the last submitted job**
    ```bash
    qstat | tail -n 1
    ```
    Displays the last submitted job's status.

11. **Submit a job to the queue**
    ```bash
    qsub script.pbs
    ```
    Submits a job script to the queue for execution.

12. **Monitor a specific job's output**
    ```bash
    tail -f job_runfinetune.o*
    ```
    Continuously view the output of a specific job.

13. **Show detailed job usage**
    ```bash
    qacct -j job_id
    ```
    Provides accounting information for a completed job.

14. **Change the priority of a job**
    ```bash
    qalter -p priority job_id
    ```
    Changes the priority of a specified job.

15. **View user's job statistics**
    ```bash
    qstat -u your_username -s
    ```
    Displays user job statistics (running, queued, etc.).

16. **Cancel all your jobs**
    ```bash
    qdel -u your_username
    ```
    Deletes all jobs submitted by your user.

17. **Check the available queues**
    ```bash
    qstat -Q
    ```
    Lists all queues available in the system.

18. **Show usage of system resources**
    ```bash
    pbsnodes -s
    ```
    Shows the current status of all nodes and their resource usage.

19. **Check available licenses**
    ```bash
    ml license_name
    ```
    Displays the available licenses for specific software.

20. **Get help on qsub options**
    ```bash
    man qsub
    ```
    Displays the manual page for the qsub command, detailing its options.
```

### Usage Instructions
- Replace `your_username`, `job_id`, `queue_name`, and `script.pbs` with the relevant values as needed.
